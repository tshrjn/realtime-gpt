import Cocoa
import Vision
import SwiftUI
import HotKey
import OpenAI

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var hotKey: HotKey!
    // OpenAI API Token from environment variable
    let openAI = OpenAI(apiToken: "YOUR_OPENAI_API_TOKEN")
    let systemPrompt = """
    As you embark on your journey as a language model, 
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

    """

    func applicationDidFinishLaunching(_ aNotification: Notification) {
        print("Application launched")
        
        hotKey = HotKey(key: .l, modifiers: [.command, .shift])
        hotKey.keyDownHandler = { [weak self] in
            print("Hotkey triggered")
            self?.captureScreen()
        }
    }

    func applicationWillTerminate(_ aNotification: Notification) {
        hotKey.keyDownHandler = nil
    }

    func captureScreen() {
        print("Capturing screen")
        
        let task = Process()
        task.launchPath = "/usr/sbin/screencapture"
        task.arguments = ["-i", "-c"]
        task.launch()
        task.waitUntilExit()

        if let image = NSPasteboard.general.readObjects(forClasses: [NSImage.self], options: nil)?.first as? NSImage {
            print("Image captured")
            getTextFromImage(image: image)
        } else {
            print("No image captured")
        }
    }

    func getTextFromImage(image: NSImage) {
        print("Extracting text from image")
        
        let requestHandler = VNImageRequestHandler(cgImage: image.cgImage(forProposedRect: nil, context: nil, hints: nil)!, options: [:])
        let request = VNRecognizeTextRequest { [weak self] request, error in
            if let error = error {
                print("Error: \(error.localizedDescription)")
                return
            }

            if let results = request.results as? [VNRecognizedTextObservation] {
                let text = results.compactMap { $0.topCandidates(1).first?.string }.joined(separator: " ")
                print("Text extracted: \(text)")
                self?.sendTextToOpenAI(text: text)
            } else {
                print("No text extracted")
            }
        }

        request.recognitionLevel = .accurate
        try? requestHandler.perform([request])
    }

    func sendTextToOpenAI(text: String) {
        print("Sending text to OpenAI Chat API")
        
        let query = ChatQuery(model: .gpt3_5Turbo, messages: [    
            .init(role: .system, content: systemPrompt),
            .init(role: .user, content: text)
            ])
        Task {
            do {
                let result = try await openAI.chats(query: query)
                // Print only the content from the response
                print("OpenAI Chat API response: \(result.choices[0].message.content)")
                // print("OpenAI Chat API response: \(result)")
            } catch {
                print("Error sending text to OpenAI Chat API: \(error.localizedDescription)")
            }
        }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()