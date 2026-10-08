// Wire format shared with the Python client. Mirrors protocol/PROTOCOL.md and
// gdrl/env/protocol.py. Change all three together and bump kVersion.
//
// This header has no Geode dependency on purpose: the standalone test server
// in mod/tools builds it with a plain C++ compiler.
#pragma once

#include <cstdint>

namespace gdrl::proto {

constexpr uint32_t kMagic = 0x4C524447;  // bytes "GDRL" read as little-endian
constexpr uint16_t kVersion = 1;
constexpr uint16_t kMaxObjects = 32;
constexpr uint16_t kDefaultPort = 5555;
constexpr uint16_t kDefaultTickRate = 240;

enum class MsgType : uint16_t {
    Hello = 1,
    HelloAck = 2,
    Reset = 3,
    Step = 4,
    State = 5,
    Error = 6,
    Bye = 7,
};

enum StateFlags : uint32_t {
    kDead = 1u << 0,
    kCompleted = 1u << 1,
    kOnGround = 1u << 2,
    kUpsideDown = 1u << 3,
    kHolding = 1u << 4,
};

enum class ObjectKind : uint8_t {
    None = 0,
    Solid = 1,
    Hazard = 2,
    Orb = 3,
    Pad = 4,
    Portal = 5,
    Other = 255,
};

enum class PlayerMode : uint8_t {
    Cube = 0,
    Ship = 1,
    Ball = 2,
    Ufo = 3,
    Wave = 4,
    Robot = 5,
    Spider = 6,
    Swing = 7,
};

enum class ErrorCode : uint16_t {
    VersionMismatch = 1,
    BadMessage = 2,
    NotInLevel = 3,
    Unsupported = 4,
};

#pragma pack(push, 1)

struct Header {
    uint32_t magic;
    uint16_t version;
    uint16_t type;    // MsgType
    uint32_t length;  // payload bytes after the header
};

struct Hello {
    uint16_t clientVersion;
    uint16_t reserved;
};

struct HelloAck {
    uint16_t protocolVersion;
    uint16_t tickRate;
    uint16_t maxObjects;
    uint16_t reserved;
};

struct Reset {
    int32_t levelId;   // 0 = current level
    int32_t startPos;  // -1 = level start
};

struct Step {
    uint8_t button;    // 0 released, 1 held
    uint8_t reserved;
    uint16_t ticks;
};

struct ObjectInfo {
    float x;  // hitbox centre
    float y;
    float w;
    float h;
    uint8_t kind;  // ObjectKind
    uint8_t reserved[3];
};

struct State {
    uint32_t tick;
    uint32_t flags;  // StateFlags
    float playerX;   // hitbox centre, game units
    float playerY;
    float velY;      // units per second, + is up
    float rotation;  // degrees
    float xSpeed;    // units per second
    float levelLength;
    uint8_t mode;    // PlayerMode
    uint8_t reserved;
    uint16_t numObjects;
    ObjectInfo objects[kMaxObjects];
};

struct Error {
    uint16_t code;  // ErrorCode
    uint16_t reserved;
    char message[64];
};

#pragma pack(pop)

// If any of these fail, the Python side will decode garbage. Fix the struct,
// not the assert.
static_assert(sizeof(Header) == 12, "Header must be 12 bytes");
static_assert(sizeof(Hello) == 4, "Hello must be 4 bytes");
static_assert(sizeof(HelloAck) == 8, "HelloAck must be 8 bytes");
static_assert(sizeof(Reset) == 8, "Reset must be 8 bytes");
static_assert(sizeof(Step) == 4, "Step must be 4 bytes");
static_assert(sizeof(ObjectInfo) == 20, "ObjectInfo must be 20 bytes");
static_assert(sizeof(State) == 676, "State must be 676 bytes");
static_assert(sizeof(Error) == 68, "Error must be 68 bytes");
static_assert(sizeof(float) == 4, "float must be IEEE-754 binary32");

}  // namespace gdrl::proto
