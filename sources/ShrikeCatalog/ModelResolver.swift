import Foundation

public enum ModelResolutionError: Error, Equatable, CustomStringConvertible {
    case unknownModel(name: String, available: [String])
    case noDefault(available: [String])
    case emptyCatalog(directory: String)

    public var description: String {
        switch self {
        case .unknownModel(let name, let available):
            return "no model named \(name); the models directory holds "
                + available.joined(separator: ", ")
        case .noDefault(let available):
            return "--model is required: the models directory holds "
                + available.joined(separator: ", ")
                + ". Name one, or mark a default in the config file."
        case .emptyCatalog(let directory):
            return "--model is required: no servable model bundles in \(directory)"
        }
    }
}

/// Resolves what to load from a flag, the configuration file and the models
/// directory, in that order.
public enum ModelResolver {
    public static let defaultConfigPath = "~/.shrike/config.json"
    public static let defaultModelsDirectory = "~/shrike-runtime/models"

    public static func loadConfig(path: String?) throws -> ShrikeConfig {
        if let path { return try ShrikeConfig.load(path: path) }
        let expanded = (defaultConfigPath as NSString).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: expanded) else { return ShrikeConfig() }
        return try ShrikeConfig.load(path: expanded)
    }

    public static func modelsDirectory(flag: String?, config: ShrikeConfig) -> URL {
        let path = flag ?? config.modelsDir ?? defaultModelsDirectory
        return URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL
    }

    /// A `--model` that names an existing directory is taken as a path; anything
    /// else is an id resolved against the models directory.
    public static func resolve(requested: String?,
                               configPath: String? = nil,
                               modelsDir: String? = nil) throws -> URL {
        if let requested, isDirectory(requested) {
            return URL(fileURLWithPath: requested).standardizedFileURL
        }
        let config = try loadConfig(path: configPath)
        let directory = modelsDirectory(flag: modelsDir, config: config)
        let scan = try ModelRoster.scanBundles(in: directory)
        let roster: ModelRoster
        do {
            roster = try ModelRoster.resolve(candidates: scan.candidates,
                                             overrides: config.models)
        } catch ModelRosterError.emptyRoster {
            // The roster's own "no servable model bundles found" cannot name the
            // directory it scanned, which is the one thing a first run needs.
            throw ModelResolutionError.emptyCatalog(directory: directory.path)
        }
        return try resolve(requested: requested, in: roster)
    }

    public static func resolve(requested: String?, in roster: ModelRoster) throws -> URL {
        guard let requested else {
            guard let defaultID = roster.defaultID else {
                throw ModelResolutionError.noDefault(available: roster.ids)
            }
            return try location(of: defaultID, in: roster)
        }
        guard let canonical = roster.canonicalID(for: requested) else {
            throw ModelResolutionError.unknownModel(name: requested, available: roster.ids)
        }
        return try location(of: canonical, in: roster)
    }

    private static func location(of id: String, in roster: ModelRoster) throws -> URL {
        guard let entry = roster.entry(for: id) else {
            throw ModelResolutionError.unknownModel(name: id, available: roster.ids)
        }
        return entry.directory
    }

    private static func isDirectory(_ path: String) -> Bool {
        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: path, isDirectory: &isDir)
        return exists && isDir.boolValue
    }
}
