import Foundation
import Shrike

/// A bundle found on disk, identified from its manifest alone — no weights
/// mapped, no Metal device. Separated from `ModelRoster.resolve` so resolution
/// is testable without a filesystem.
public struct RosterCandidate: Sendable, Equatable {
    public let bundleName: String
    public let directory: URL
    public let manifestModelID: String
    public let family: ModelFamily

    public init(bundleName: String, directory: URL, manifestModelID: String, family: ModelFamily) {
        self.bundleName = bundleName
        self.directory = directory
        self.manifestModelID = manifestModelID
        self.family = family
    }
}

/// One servable model under its canonical API id.
public struct RosterEntry: Sendable, Equatable {
    public let id: String
    public let bundleName: String
    public let directory: URL
    public let manifestModelID: String
    public let family: ModelFamily

    public init(id: String, bundleName: String, directory: URL,
                manifestModelID: String, family: ModelFamily) {
        self.id = id
        self.bundleName = bundleName
        self.directory = directory
        self.manifestModelID = manifestModelID
        self.family = family
    }
}

public enum ModelRosterError: Error, Equatable, CustomStringConvertible {
    case emptyRoster
    case duplicateID(id: String, claimants: [String])
    case unknownOverrideDir(String)
    case overrideNotServable(dir: String, family: String)
    case modelsDirUnreadable(path: String, reason: String)

    public var description: String {
        switch self {
        case .emptyRoster:
            return "no servable model bundles found"
        case .duplicateID(let id, let claimants):
            return "model id \(id) is claimed by \(claimants.joined(separator: " and "))"
        case .unknownOverrideDir(let dir):
            return "config names \(dir), which matches no bundle"
        case .overrideNotServable(let dir, let family):
            return "config names \(dir), whose family \(family) is not servable"
        case .modelsDirUnreadable(let path, let reason):
            return "cannot scan models directory \(path): \(reason)"
        }
    }
}

/// The resolved model namespace: every servable bundle under its canonical id,
/// with bundle names accepted as aliases in the same namespace, and which id
/// serves a request that names no model.
public struct ModelRoster: Sendable, Equatable {
    public let entries: [RosterEntry]
    public let defaultID: String?
    /// Bundles whose family is not servable, dropped by rule — logged at
    /// startup, never an error.
    public let droppedBundles: [String]

    private let namespace: [String: String]
    private let entriesByID: [String: RosterEntry]

    public var ids: [String] { entries.map(\.id) }

    public func canonicalID(for name: String) -> String? { namespace[name] }

    public func entry(for name: String) -> RosterEntry? {
        namespace[name].flatMap { entriesByID[$0] }
    }

    static func isServable(_ family: ModelFamily) -> Bool { family != .qwen36MTP }

    public static func resolve(candidates: [RosterCandidate],
                               overrides: [ShrikeConfig.ModelOverride]) throws -> ModelRoster {
        let servable = candidates.filter { isServable($0.family) }
        let dropped = candidates.filter { !isServable($0.family) }.map(\.bundleName).sorted()

        var overrideByBundle: [String: ShrikeConfig.ModelOverride] = [:]
        for override in overrides {
            let bundleName = override.dir.hasSuffix(".gturbo")
                ? String(override.dir.dropLast(".gturbo".count))
                : override.dir
            guard let candidate = candidates.first(where: { $0.bundleName == bundleName }) else {
                throw ModelRosterError.unknownOverrideDir(override.dir)
            }
            guard isServable(candidate.family) else {
                throw ModelRosterError.overrideNotServable(
                    dir: override.dir, family: candidate.family.rawValue)
            }
            overrideByBundle[bundleName] = override
        }

        var entries: [RosterEntry] = []
        var defaultID: String?
        for candidate in servable {
            let override = overrideByBundle[candidate.bundleName]
            let id = override?.id ?? candidate.bundleName
            entries.append(RosterEntry(id: id,
                                       bundleName: candidate.bundleName,
                                       directory: candidate.directory,
                                       manifestModelID: candidate.manifestModelID,
                                       family: candidate.family))
            if override?.isDefault == true { defaultID = id }
        }
        guard !entries.isEmpty else { throw ModelRosterError.emptyRoster }
        entries.sort { $0.id < $1.id }

        // Ids and bundle names share one namespace, so a config id colliding
        // with another bundle's name is a duplicate, not a shadow.
        var claims: [String: [String]] = [:]
        for entry in entries {
            claims[entry.id, default: []].append(entry.bundleName)
            if entry.bundleName != entry.id {
                claims[entry.bundleName, default: []].append(entry.bundleName)
            }
        }
        for name in claims.keys.sorted() {
            let claimants = claims[name, default: []]
            if claimants.count > 1 {
                throw ModelRosterError.duplicateID(id: name, claimants: claimants.sorted())
            }
        }

        var namespace: [String: String] = [:]
        var entriesByID: [String: RosterEntry] = [:]
        for entry in entries {
            namespace[entry.id] = entry.id
            namespace[entry.bundleName] = entry.id
            entriesByID[entry.id] = entry
        }
        if defaultID == nil, entries.count == 1 { defaultID = entries[0].id }

        return ModelRoster(entries: entries, defaultID: defaultID, droppedBundles: dropped,
                           namespace: namespace, entriesByID: entriesByID)
    }

    private init(entries: [RosterEntry], defaultID: String?, droppedBundles: [String],
                 namespace: [String: String], entriesByID: [String: RosterEntry]) {
        self.entries = entries
        self.defaultID = defaultID
        self.droppedBundles = droppedBundles
        self.namespace = namespace
        self.entriesByID = entriesByID
    }
}

extension ModelRoster {
    public struct SkippedBundle: Sendable, Equatable {
        public let bundleName: String
        public let reason: String

        public init(bundleName: String, reason: String) {
            self.bundleName = bundleName
            self.reason = reason
        }
    }

    public struct BundleScan: Sendable, Equatable {
        public let candidates: [RosterCandidate]
        /// Bundles whose manifest could not be read or names an unknown
        /// family — reported so the caller can log them, never fatal: one
        /// unreadable bundle must not take down the roster.
        public let skipped: [SkippedBundle]

        public init(candidates: [RosterCandidate], skipped: [SkippedBundle]) {
            self.candidates = candidates
            self.skipped = skipped
        }
    }

    /// Roster for `--model`: exactly this bundle, ignoring any config. Its
    /// canonical id is the bundle's directory name unless one is given.
    public static func single(directory: URL,
                              overrideID: String? = nil) throws -> ModelRoster {
        let identity = try ManifestReader.peekIdentity(directoryURL: directory)
        let name = directory.lastPathComponent
        let bundleName = name.hasSuffix(".gturbo")
            ? String(name.dropLast(".gturbo".count)) : name
        let candidate = RosterCandidate(bundleName: bundleName,
                                        directory: directory,
                                        manifestModelID: identity.modelID,
                                        family: identity.family)
        let overrides = overrideID.map {
            [ShrikeConfig.ModelOverride(dir: bundleName, id: $0, isDefault: true)]
        } ?? []
        return try resolve(candidates: [candidate], overrides: overrides)
    }

    public static func scanBundles(in directory: URL) throws -> BundleScan {
        let contents: [URL]
        do {
            contents = try FileManager.default.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
        } catch {
            throw ModelRosterError.modelsDirUnreadable(
                path: directory.path, reason: error.localizedDescription)
        }

        var candidates: [RosterCandidate] = []
        var skipped: [ModelRoster.SkippedBundle] = []
        let bundles = contents
            .filter { $0.pathExtension == "gturbo" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        for url in bundles {
            let isDirectory = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
            guard isDirectory else { continue }
            let name = url.deletingPathExtension().lastPathComponent
            do {
                let identity = try ManifestReader.peekIdentity(directoryURL: url)
                candidates.append(RosterCandidate(bundleName: name,
                                                  directory: url,
                                                  manifestModelID: identity.modelID,
                                                  family: identity.family))
            } catch {
                skipped.append(SkippedBundle(bundleName: name, reason: String(describing: error)))
            }
        }
        return BundleScan(candidates: candidates, skipped: skipped)
    }
}
