// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "Shrike",
    platforms: [
        .macOS(.v26),
    ],
    products: [
        .library(name: "Shrike", targets: ["Shrike"]),
        .library(name: "ShrikeFormat", targets: ["ShrikeFormat"]),
        .executable(name: "shrike", targets: ["ShrikeRoot"]),
        .executable(name: "shrike-bench", targets: ["ShrikeBench"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.3.6"),
        .package(url: "https://github.com/apple/swift-collections.git", from: "1.5.1"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.99.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.8.2"),
    ],
    targets: [
        .target(
            name: "ShrikeFormat",
            path: "sources/ShrikeFormat"
        ),
        // C99 + NEON for the inner loops where Swift's vector types do not
        // lower well. Kept deliberately small: one file, one entry point,
        // covered by the same tests as the Swift path it replaced. No custom
        // flags -- -O3 measured the same as SwiftPM's release default (0.675
        // vs 0.680 ms), so it is not worth the unsafeFlags constraint.
        .target(
            name: "ShrikeKernelsC",
            path: "sources/ShrikeKernelsC"
        ),
        .target(
            name: "Shrike",
            dependencies: [
                "ShrikeFormat",
                "ShrikeKernelsC",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Jinja", package: "swift-jinja"),
                .product(name: "OrderedCollections", package: "swift-collections"),
            ],
            path: "sources/Shrike",
            resources: [
                .copy("Metal"),
            ]
        ),
        .target(
            name: "ShrikeCatalog",
            dependencies: ["Shrike"],
            path: "sources/ShrikeCatalog"
        ),
        .target(
            name: "ShrikeArgumentSupport",
            dependencies: [
                "Shrike",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "sources/ShrikeArgumentSupport"
        ),
        .target(
            name: "ShrikeRepackCore",
            dependencies: [
                "ShrikeFormat",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "sources/ShrikeRepack/Core"
        ),
        .target(
            name: "ShrikeCLICore",
            dependencies: [
                "Shrike",
                "ShrikeCatalog",
                "ShrikeArgumentSupport",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "sources/ShrikeCLI"
        ),
        .target(
            name: "ShrikeAttnBenchCore",
            dependencies: [
                "Shrike",
                "ShrikeValidationSupport",
                "ShrikeArgumentSupport",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "sources/ShrikeAttnBench",
            resources: [
                .copy("Metal"),
            ]
        ),
        .target(
            name: "ShrikeExpertBenchCore",
            dependencies: [
                "Shrike",
                "ShrikeArgumentSupport",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "sources/ShrikeExpertBench",
            resources: [
                .copy("Metal"),
            ]
        ),
        .target(
            name: "ShrikeServerCore",
            dependencies: [
                "Shrike",
                "ShrikeCatalog",
                "ShrikeArgumentSupport",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "sources/ShrikeServer/Core"
        ),
        .target(
            name: "ShrikeRootCore",
            dependencies: [
                "ShrikeCLICore",
                "ShrikeServerCore",
                "ShrikeRepackCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "sources/ShrikeRoot/Core"
        ),
        .executableTarget(
            name: "ShrikeRoot",
            dependencies: ["ShrikeRootCore"],
            path: "sources/ShrikeRoot/Command"
        ),
        .target(
            name: "ShrikeBenchCore",
            dependencies: [
                "ShrikeAttnBenchCore",
                "ShrikeExpertBenchCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "sources/ShrikeBench/Core"
        ),
        .executableTarget(
            name: "ShrikeBench",
            dependencies: ["ShrikeBenchCore"],
            path: "sources/ShrikeBench/Command"
        ),
        .target(
            name: "ShrikeValidationSupport",
            dependencies: ["Shrike"],
            path: "sources/ShrikeValidation/Support"
        ),
        .testTarget(
            name: "ShrikeTestsCore",
            dependencies: [
                "Shrike", "ShrikeKernelsC", "ShrikeValidationSupport", "ShrikeRepackCore", "ShrikeCLICore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "tests/Shrike/Core",
            resources: [.copy("Tokenization/Fixtures")]
        ),
        .testTarget(
            name: "ShrikeRepackTests",
            dependencies: [
                "ShrikeRepackCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "tests/ShrikeRepack/Core"
        ),
        .testTarget(
            name: "ShrikeBenchTests",
            dependencies: [
                "ShrikeAttnBenchCore",
                "ShrikeExpertBenchCore",
                "ShrikeBenchCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "tests/ShrikeBench"
        ),
        .testTarget(
            name: "ShrikeCatalogTests",
            dependencies: ["ShrikeCatalog", "Shrike"],
            path: "tests/ShrikeCatalog"
        ),
        .testTarget(
            name: "ShrikeRootTests",
            dependencies: [
                "ShrikeRootCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "tests/ShrikeRoot"
        ),
        .testTarget(
            name: "ShrikeServerTests",
            dependencies: [
                "ShrikeServerCore",
                "ShrikeCatalog",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "tests/ShrikeServer",
            resources: [.copy("Fixtures")]
        ),
    ],
    swiftLanguageModes: [.v6]
)
