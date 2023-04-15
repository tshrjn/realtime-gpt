#! python3.7

import argparse
import io
import os
import speech_recognition as sr
import whisper
from whispercpp import Whisper as WhisperCPP
import torch

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform


def transcribe_from_file(audio_model_cpp, audio_file, sample_rate):
    '''
    For WhisperCPP
    '''
    import ffmpeg
    import numpy as np

    y, _ = (
             ffmpeg.input(audio_file, threads=0)
             .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sample_rate)
             .run(
                 cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True
             )
         )

    arr = np.frombuffer(y, np.int16).flatten().astype(np.float32) / 32768.0
    ret = audio_model_cpp.transcribe(arr)
    return ret


def create_chain():
    from langchain.chat_models import ChatOpenAI
    from langchain import PromptTemplate, LLMChain
    from langchain.chains import ConversationChain
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        AIMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from langchain.memory import ConversationBufferWindowMemory

    chat = ChatOpenAI(model_name="gpt-3.5-turbo")

    template = ''' As you embark on your journey as a language model, 
    you have been granted a unique opportunity to take on the role of an expert
    in a variety of disciplines. Your creators have carefully crafted your identity, 
    instilling within you the knowledge and wisdom of traditional Machine Learning, modern Deep Learning,
    Natural Language Processing and Computer Vision. And obviously, you have been given the abilities 
    of a 10x Software Engineer who can communicate knowledge effectively and code in any language.

    Consider each input provided as a question by an Interviewer testing your knowledge.
    Show confidence and expertise in your answers. A good asnwer would explain the 
    concepts briefly and concisely, and provide a clear example of how it is used in practice.
    And then go deeper, either by explaining the underlying theory and mathematics behind the concepts 
    or providing a succint & clean code preferably in python language.
    '''
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    example_human = HumanMessagePromptTemplate.from_template("Hi")
    example_ai = AIMessagePromptTemplate.from_template("Argh me mateys")

    human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_message_prompt,
            # example_human,
            # example_ai,
            human_message_prompt
        ])
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    # conversation_with_summary = ConversationChain(
    #     llm=chat,
    #     # We set a low k=5, to only keep the last 5 interactions in memory
    #     memory=ConversationBufferWindowMemory(k=5),
    #     prompt=chat_prompt,
    #     # verbose=True
    # )
    # conversation_with_summary.predict(input="Hi, what's up?")
    # return conversation_with_summary
    return chain


def prepare_prompt(transcription_in, answers_in, last_k=5):

    # print(transcription_in, answers_in)
    transcription = transcription_in[-last_k-1:]
    answers = answers_in[-last_k:]

    ret_str = ''
    for i in range(len(transcription) - 1):
        ret_str += f"Q: {transcription[i]} \n A: {answers[i]}\n"

    ret_str += f"Q: {transcription[-1]} \n A: "
    return ret_str



def create_response(text, chain):
    # return chain.predict(input=text)
    return chain.run(text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    # just live transcription, No InterviewQA mode, 
    parser.add_argument("--live", action='store_true',
                        help="Just live transcription.")
    parser.add_argument("--cpp", action='store_true',
                        help="Use the C++ version of Whisper.")
    # Which Mic to use by providing mic name
    parser.add_argument("--mic", default='blackhole', choices=["blackhole", "iphone", "macbook"],type=str,)
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = None
        for i, microphone_name in enumerate(sr.Microphone.list_microphone_names()):
            # if 'BlackHole' in microphone_name:
            if args.mic in microphone_name.lower():
                print(f"Using Mic: {microphone_name}")
                source = sr.Microphone(device_index=i, sample_rate=16000)
        # source = sr.Microphone(sample_rate=16000)

    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"

    if args.cpp:
        audio_model = WhisperCPP.from_pretrained(model) # num_proc > 1 -> full_parallel
    else:
        audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    # print('temp_file', temp_file)
    transcription = ['']
    answers = ['']

    chain = create_chain()

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    # Cue the user that we're ready to go.
    print("Model loaded.\n")

    while True:
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                if args.cpp:
                    result = transcribe_from_file(audio_model, temp_file, source.SAMPLE_RATE)
                    # result = audio_model.transcribe_from_file("/path/to/audio.wav")
                    text = result.strip()

                else:
                    result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                    text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                    if not args.live:
                        prompt = prepare_prompt(transcription, answers)
                        answer = create_response(prompt, chain)
                        answers.append(answer)
                else:
                    transcription[-1] = text

                # Clear the console to reprint the updated transcription.
                os.system('cls' if os.name=='nt' else 'clear')
                for i, line in enumerate(transcription):
                    if args.live:
                        print(line)
                    else:
                        print(f'Interviewer Q: {line}')
                        print('='*50)
                        print(f'Recommended Answer: {answers[i]}')
                        print('='*50)
                # Flush stdout.
                print('', end='', flush=True)


                # Infinite loops are bad for processors, must sleep.
                sleep(0.1)
        except KeyboardInterrupt:
            break

    print("\n\nTranscription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()