import Foundation
import Vision

if CommandLine.arguments.count != 2 { fatalError("Expected one local image path") }
let url = URL(fileURLWithPath: CommandLine.arguments[1])
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false
request.automaticallyDetectsLanguage = true
request.recognitionLanguages = try request.supportedRecognitionLanguages()
let handler = VNImageRequestHandler(url: url, options: [:])
do {
    try handler.perform([request])
    let observations: [[String: Any]] = (request.results ?? []).map { observation in
        let box = observation.boundingBox
        let candidate = observation.topCandidates(1).first
        return ["text": candidate?.string ?? "", "confidence": candidate?.confidence ?? 0,
                "box": [box.origin.x, box.origin.y, box.width, box.height]]
    }
    let payload: [String: Any] = ["observations": observations, "canonical_success": false]
    let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    print(String(decoding: data, as: UTF8.self))
} catch {
    fputs("Vision error: \(error)\n", stderr)
    exit(1)
}
