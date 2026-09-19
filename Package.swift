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
        .executable(name: "ShrikeRepack", targets: ["ShrikeRepack"]),
        .executable(name: "ShrikeCLI", targets: ["ShrikeCLI"]),
        .executable(name: "ShrikeAttnBench", targets: ["ShrikeAttnBench"]),
        .executable(name: "ShrikeExpertBench", targets: ["ShrikeExpertBench"]),
        .executable(name: "ShrikeMac", targets: ["ShrikeMac"]),
        .executable(name: "ShrikeDecodeService", targets: ["ShrikeDecodeService"]),
        .executable(name: "ShrikeServer", targets: ["ShrikeServer"]),
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
            name: "ShrikeRepackCore",
            dependencies: ["ShrikeFormat"],
            path: "sources/ShrikeRepack/Core"
        ),
        .executableTarget(
            name: "ShrikeRepack",
            dependencies: ["ShrikeRepackCore"],
            path: "sources/ShrikeRepack/Command"
        ),
        .target(
            name: "ShrikeCLICore",
            dependencies: [
                "Shrike",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "sources/ShrikeCLI",
            exclude: ["Command"]
        ),
        .executableTarget(
            name: "ShrikeCLI",
            dependencies: ["ShrikeCLICore"],
            path: "sources/ShrikeCLI/Command"
        ),
        .executableTarget(
            name: "ShrikeAttnBench",
            dependencies: ["Shrike", "ShrikeValidationSupport"],
            path: "sources/ShrikeAttnBench",
            resources: [
                .copy("Metal"),
            ]
        ),
        .executableTarget(
            name: "ShrikeExpertBench",
            dependencies: ["Shrike"],
            path: "sources/ShrikeExpertBench",
            resources: [
                .copy("Metal"),
            ]
        ),
        .target(
            name: "ShrikeAppCore",
            dependencies: ["Shrike", "ShrikeRepackCore", "ShrikeDecodeProtocol"],
            path: "sources/ShrikeApp/Core",
            resources: [
                .copy("Resources/app-prompts.json"),
            ]
        ),
        .target(
            name: "ShrikeMacPresentation",
            dependencies: ["ShrikeAppCore"],
            path: "sources/ShrikeApp/MacPresentation"
        ),
        .target(
            name: "ShrikeDecodeProtocol",
            path: "sources/ShrikeDecodeProtocol"
        ),
        .executableTarget(
            name: "ShrikeDecodeService",
            dependencies: ["ShrikeAppCore", "ShrikeDecodeProtocol"],
            path: "sources/ShrikeDecodeService"
        ),
        .target(
            name: "ShrikeServerCore",
            dependencies: [
                "Shrike",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "sources/ShrikeServer/Core"
        ),
        .executableTarget(
            name: "ShrikeServer",
            dependencies: ["ShrikeServerCore"],
            path: "sources/ShrikeServer/Command"
        ),
        .executableTarget(
            name: "ShrikeMac",
            dependencies: ["ShrikeAppCore", "ShrikeMacPresentation"],
            path: "sources/ShrikeApp/Mac",
            resources: [
                .copy("Resources/shrike-app-icon.png"),
            ]
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
            dependencies: ["ShrikeRepackCore"],
            path: "tests/ShrikeRepack/Core"
        ),
        .testTarget(
            name: "ShrikeAppCoreTests",
            dependencies: ["ShrikeAppCore", "Shrike", "ShrikeRepackCore", "ShrikeDecodeProtocol"],
            path: "tests/ShrikeApp/Core"
        ),
        .testTarget(
            name: "ShrikeDecodeServiceTests",
            dependencies: ["ShrikeDecodeService", "ShrikeAppCore", "ShrikeDecodeProtocol"],
            path: "tests/ShrikeDecodeService"
        ),
        .testTarget(
            name: "ShrikeMacPresentationTests",
            dependencies: ["ShrikeAppCore", "ShrikeMacPresentation"],
            path: "tests/ShrikeApp/MacPresentation"
        ),
        .testTarget(
            name: "ShrikeServerTests",
            dependencies: [
                "ShrikeServerCore",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "tests/ShrikeServer",
            resources: [.copy("Fixtures")]
        ),
    ],
    swiftLanguageModes: [.v6]
)
