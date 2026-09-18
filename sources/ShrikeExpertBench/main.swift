import Foundation

let benchArgs: BenchArgs
do {
    benchArgs = try BenchArgs.parse(Array(CommandLine.arguments.dropFirst()))
} catch BenchError.help {
    print(BenchArgs.usage)
    exit(0)
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n\n\(BenchArgs.usage)\n".utf8))
    exit(2)
}

do {
    let runner = try BenchRunner(args: benchArgs)
    try runner.run()
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}
