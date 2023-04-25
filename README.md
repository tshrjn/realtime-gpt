# Real-Time GPT 

![Demo gif](demo.gif)

This is a demo of real time speech to text with OpenAI's Whisper model. It works by constantly recording audio in a thread and concatenating the raw bytes over multiple recordings.

### Usage

```
python demo.py
--cpp: Use WhisperCPP
--model: Select whisper model size
--live: Live Transcription Only
--mic: Which Microphone to use
```
I recommend using Blackhole to route audio to the model.


### Installation
#### Screenshot to Answering
* For Screenshot to Answer:
Use `swift build` and then `swift run` to run the program.


#### Live Transcription & Answering
To install dependencies simply run
```
pip install -r requirements.txt
```
in an environment of your choosing.

Whisper also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

For more information on Whisper please see https://github.com/openai/whisper

The code in this repository is public domain.

### TODO:
- [ ] Combine Visual and Audio background
- [ ] Start and stop from keyboard shortcut
- [ ] A small GUI to control settings
- [ ] Add STT model options to use whisper API, native MacOS
