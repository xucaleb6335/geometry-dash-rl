// Wire format shared with the Python client. Mirrors protocol/PROTOCOL.md and
// gdrl/env/protocol.py. Change all three together and bump kVersion.
//
// This header has no Geode dependency on purpose: the standalone test server
// in mod/tools builds it with a plain C++ compiler.
//
// Versions: the server speaks 1 and 2 and answers in the version the client's
// HELLO asked for. v2 adds snapshots (SAVE/LOAD/DROP_STATE), StateV2 (player 2,
// hitbox size, floor/ceiling, 64 objects) and object subtypes/flags.
#pragma once

#include <cstdint>

namespace gdrl::proto {

constexpr uint32_t kMagic = 0x4C524447;  // bytes "GDRL" read as little-endian
constexpr uint16_t kVersion = 2;         // newest version spoken
constexpr uint16_t kMinVersion = 1;
constexpr uint16_t kMaxObjects = 32;     // v1 STATE
constexpr uint16_t kMaxObjectsV2 = 64;   // v2 STATE
constexpr uint16_t kMaxSnapshots = 256;  // v2 snapshot slots
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
    SaveState = 8,  // v2
    LoadState = 9,  // v2
    DropState = 10, // v2
};

enum StateFlags : uint32_t {
    kDead = 1u << 0,
    kCompleted = 1u << 1,
    kOnGround = 1u << 2,
    kUpsideDown = 1u << 3,
    kHolding = 1u << 4,
    kMini = 1u << 5,  // v2
    kDual = 1u << 6,  // v2
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

// v2: what exactly an object is. Same values as gdrl/env/protocol.py.
enum class ObjectSubtype : uint8_t {
    Unknown = 0,
    Block = 1, Slope = 2, Spike = 3, Saw = 4,
    OrbYellow = 10, OrbPink = 11, OrbGravity = 12, OrbGreen = 13, OrbDrop = 14, OrbRed = 15,
    OrbDash = 16, OrbGravityDash = 17, OrbSpider = 18, OrbTeleport = 19, OrbCustom = 20,
    PadYellow = 30, PadPink = 31, PadGravity = 32, PadRed = 33, PadSpider = 34,
    PortalCube = 40, PortalShip = 41, PortalBall = 42, PortalUfo = 43, PortalWave = 44,
    PortalRobot = 45, PortalSpider = 46, PortalSwing = 47,
    PortalGravityDown = 50, PortalGravityUp = 51, PortalGravityToggle = 52, PortalMini = 53,
    PortalNormalSize = 54, PortalDual = 55, PortalSolo = 56, PortalMirror = 57, PortalMirrorOff = 58,
    PortalTeleport = 59,
    SpeedSlow = 60, SpeedNormal = 61, Speed2x = 62, Speed3x = 63, Speed4x = 64,
};

enum ObjectFlags : uint8_t {
    kObjMoved = 1u << 0,      // not at its spawn position
    kObjActivated = 1u << 1,  // orb/pad/portal already used this attempt
    kObjHidden = 1u << 2,     // invisible but still collides
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
    NoSnapshot = 5,  // v2
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
    uint16_t maxSnapshots;  // v1: reserved (0)
};

struct Reset {
    int32_t levelId;   // 0 = current level
    int32_t startPos;  // -1 = level start, k >= 0 = k-th Start Pos by x
};

struct Step {
    uint8_t button;    // 0 released, 1 held
    uint8_t reserved;
    uint16_t ticks;
};

struct Snapshot {  // v2: SAVE_STATE, LOAD_STATE, DROP_STATE
    int32_t slot;
    uint32_t reserved;
};

struct ObjectInfo {
    float x;  // hitbox centre
    float y;
    float w;
    float h;
    uint8_t kind;     // ObjectKind
    uint8_t subtype;  // v2 ObjectSubtype (0 in v1)
    uint8_t flags;    // v2 ObjectFlags (0 in v1)
    uint8_t reserved;
};

struct State {  // v1
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

struct StateV2 {
    uint32_t tick;
    uint32_t flags;
    float playerX;
    float playerY;
    float velY;
    float rotation;
    float xSpeed;
    float levelLength;
    uint8_t mode;
    uint8_t reserved;
    uint16_t numObjects;
    float playerSize;  // side of player 1's hitbox
    float floorY;      // NaN = unknown / none
    float ceilY;       // NaN = unknown / none
    float p2X;
    float p2Y;
    float p2VelY;
    uint8_t p2Mode;
    uint8_t p2Flags;   // StateFlags bits: Dead, OnGround, UpsideDown, Mini
    uint16_t reserved2;
    ObjectInfo objects[kMaxObjectsV2];
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
static_assert(sizeof(Snapshot) == 8, "Snapshot must be 8 bytes");
static_assert(sizeof(ObjectInfo) == 20, "ObjectInfo must be 20 bytes");
static_assert(sizeof(State) == 676, "State must be 676 bytes");
static_assert(sizeof(StateV2) == 1344, "StateV2 must be 1344 bytes");
static_assert(sizeof(Error) == 68, "Error must be 68 bytes");
static_assert(sizeof(float) == 4, "float must be IEEE-754 binary32");

// The v1 view of a v2 state: same fields, first 32 objects.
inline State toV1(const StateV2& s) {
    State out{};
    out.tick = s.tick;
    out.flags = s.flags & (kDead | kCompleted | kOnGround | kUpsideDown | kHolding);
    out.playerX = s.playerX;
    out.playerY = s.playerY;
    out.velY = s.velY;
    out.rotation = s.rotation;
    out.xSpeed = s.xSpeed;
    out.levelLength = s.levelLength;
    out.mode = s.mode;
    out.numObjects = s.numObjects < kMaxObjects ? s.numObjects : kMaxObjects;
    for (uint16_t i = 0; i < out.numObjects; ++i) {
        out.objects[i] = s.objects[i];
        out.objects[i].subtype = 0;
        out.objects[i].flags = 0;
    }
    return out;
}

}  // namespace gdrl::proto
