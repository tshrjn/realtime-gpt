// swift-tools-version:5.3
import PackageDescription

let package = Package(
    name: "ocr_tool",
    platforms: [
        .macOS(.v10_15)
    ],
    dependencies: [
        .package(url: "https://github.com/soffes/HotKey.git", from: "0.1.3"),
        .package(url: "https://github.com/MacPaw/OpenAI.git", from: "0.2.5"),
        // .package(url: "https://github.com/MacPaw/OpenAI.git", from: "0.1.2"),
        
    ],
    targets: [
        .target(
            name: "ocr_tool",
            dependencies: [
                .product(name: "HotKey", package: "HotKey"),
                .product(name: "OpenAI", package: "OpenAI"),
            ]),
        .testTarget(
            name: "ocr_toolTests",
            dependencies: ["ocr_tool"]),
    ]
)