import Foundation
import Vision
let url=URL(fileURLWithPath:CommandLine.arguments[1])
let request=VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false
request.automaticallyDetectsLanguage = true
request.recognitionLanguages = try request.supportedRecognitionLanguages()
try VNImageRequestHandler(url:url,options:[:]).perform([request])
let pattern=try NSRegularExpression(pattern:"\\S+")
var output:[[String:Any]]=[]
for observation in request.results ?? [] {
    guard let candidate=observation.topCandidates(1).first else {continue}
    let value=candidate.string
    for match in pattern.matches(in:value,range:NSRange(value.startIndex...,in:value)) {
        guard let range=Range(match.range,in:value) else {continue}
        let token=String(value[range])
        if !token.contains(where:{$0.isNumber}) {continue}
        guard let rectangle=try candidate.boundingBox(for:range) else {continue}
        let b=rectangle.boundingBox
        output.append(["token":token,"line":value,"box":[b.minX,b.minY,b.width,b.height]])
    }
}
let data=try JSONSerialization.data(withJSONObject:output,options:[.sortedKeys])
print(String(decoding:data,as:UTF8.self))
