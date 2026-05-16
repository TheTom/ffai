// FFAI CLI — `ffai <subcommand>`.
//
// Default subcommand is `generate`, so `ffai --model X --prompt Y`
// keeps working without typing the subcommand. `ffai bench --method
// simple --model X --prompt Y` runs a benchmark instead.

import ArgumentParser
import FFAI
import Foundation

@main
struct FFAIRoot: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ffai",
        abstract: "Fucking Fast Apple Inference — Apple Silicon LLM CLI.",
        subcommands: [GenerateCommand.self, BenchCommand.self],
        defaultSubcommand: GenerateCommand.self
    )
}
