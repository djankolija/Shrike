import Foundation
import Shrike

/// The configuration file (`--config`, default `~/.shrike/config.json`).
///
/// `models` is a list of overrides, not the roster: a bundle with no entry is
/// still served, under its bundle name. Config is written only for models whose
/// name needs fixing, plus at most one `default` marking the model served when
/// a request omits `model`. The file is read once at startup and never written
/// back.
public struct ShrikeConfig: Sendable, Equatable {
    public struct Defaults: Sendable, Equatable, Decodable {
        public let maxContext: Int?
        public let ramBudget: String?

        enum CodingKeys: String, CodingKey {
            case maxContext = "max_context"
            case ramBudget = "ram_budget"
        }

        public init(maxContext: Int? = nil, ramBudget: String? = nil) {
            self.maxContext = maxContext
            self.ramBudget = ramBudget
        }
    }

    public struct ModelOverride: Sendable, Equatable, Decodable {
        public let dir: String
        public let id: String?
        public let isDefault: Bool

        enum CodingKeys: String, CodingKey {
            case dir
            case id
            case isDefault = "default"
        }

        public init(dir: String, id: String? = nil, isDefault: Bool = false) {
            self.dir = dir
            self.id = id
            self.isDefault = isDefault
        }

        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.dir = try container.decode(String.self, forKey: .dir)
            self.id = try container.decodeIfPresent(String.self, forKey: .id)
            self.isDefault = try container.decodeIfPresent(Bool.self, forKey: .isDefault) ?? false
        }

        public var bundleName: String {
            dir.hasSuffix(".gturbo") ? String(dir.dropLast(".gturbo".count)) : dir
        }
    }

    public let modelsDir: String?
    public let defaults: Defaults
    public let models: [ModelOverride]

    public init(modelsDir: String? = nil,
                defaults: Defaults = Defaults(),
                models: [ModelOverride] = []) {
        self.modelsDir = modelsDir
        self.defaults = defaults
        self.models = models
    }
}

extension ShrikeConfig: Decodable {
    enum CodingKeys: String, CodingKey {
        case modelsDir = "models_dir"
        case defaults
        case models
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelsDir = try container.decodeIfPresent(String.self, forKey: .modelsDir)
        self.defaults = try container.decodeIfPresent(Defaults.self, forKey: .defaults) ?? Defaults()
        self.models = try container.decodeIfPresent([ModelOverride].self, forKey: .models) ?? []
    }
}

public enum ShrikeConfigError: Error, Equatable, CustomStringConvertible {
    case unreadable(path: String, reason: String)
    case invalid(String)

    public var description: String {
        switch self {
        case .unreadable(let path, let reason):
            return "cannot read config \(path): \(reason)"
        case .invalid(let reason):
            return "invalid config: \(reason)"
        }
    }
}

extension ShrikeConfig {
    public static func load(path: String) throws -> ShrikeConfig {
        let expanded = (path as NSString).expandingTildeInPath
        let data: Data
        do {
            data = try Data(contentsOf: URL(fileURLWithPath: expanded))
        } catch {
            throw ShrikeConfigError.unreadable(path: expanded, reason: error.localizedDescription)
        }
        return try parse(data)
    }

    public static func parse(_ data: Data) throws -> ShrikeConfig {
        let config: ShrikeConfig
        do {
            config = try JSONDecoder().decode(ShrikeConfig.self, from: data)
        } catch {
            throw ShrikeConfigError.invalid(String(describing: error))
        }
        try config.validate()
        return config
    }

    private func validate() throws {
        var seenDirs: Set<String> = []
        var defaultDirs: [String] = []
        for model in models {
            guard !model.dir.isEmpty else {
                throw ShrikeConfigError.invalid("a models entry has an empty dir")
            }
            if let id = model.id, id.isEmpty {
                throw ShrikeConfigError.invalid("model \(model.dir) has an empty id")
            }
            guard seenDirs.insert(model.bundleName).inserted else {
                throw ShrikeConfigError.invalid("model \(model.dir) appears twice")
            }
            if model.isDefault { defaultDirs.append(model.dir) }
        }
        if defaultDirs.count > 1 {
            throw ShrikeConfigError.invalid(
                "more than one default: \(defaultDirs.joined(separator: ", "))")
        }
        if let budget = defaults.ramBudget, RuntimeConfiguration.parseBudgetBytes(budget) == nil {
            throw ShrikeConfigError.invalid(
                "ram_budget must be a positive size such as 2G, 512M or a byte count")
        }
        if let context = defaults.maxContext, context < 1 {
            throw ShrikeConfigError.invalid("max_context must be positive")
        }
    }
}
