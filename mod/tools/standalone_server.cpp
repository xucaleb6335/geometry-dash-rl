// A headless stand-in for the mod: Bridge + Protocol with a trivial fake game,
// no Geode and no GD. It exists to prove the C++ structs and the Python decoder
// agree byte for byte (tests/test_cpp_bridge.py), and to give you a C++ server
// to poke at while the real hooks are still TODO.
//
// Fake game: the player moves right at 1x speed; y is 45 while the button is
// held and 15 otherwise; one hazard sits at x = 600. The player "dies" at tick
// 1000 so terminal handling can be tested.
//
// Build:  cmake -S mod/tools -B build/tools && cmake --build build/tools
// Run:    build/tools/standalone_server [port]   (port 0 = OS picks, printed on stdout)

#include "Bridge.hpp"

#include <cstdio>
#include <cstdlib>
#include <variant>

using namespace gdrl;

namespace {

constexpr float kSpeed = 311.58f;  // units per second at 1x speed
constexpr uint32_t kDeathTick = 1000;

proto::State makeState(uint32_t tick, bool holding) {
    proto::State s{};
    s.tick = tick;
    s.flags = holding ? proto::kHolding : 0u;
    if (!holding) s.flags |= proto::kOnGround;
    if (tick >= kDeathTick) s.flags |= proto::kDead;
    s.playerX = 15.0f + kSpeed * static_cast<float>(tick) / proto::kDefaultTickRate;
    s.playerY = holding ? 45.0f : 15.0f;
    s.velY = 0.0f;
    s.xSpeed = kSpeed;
    s.levelLength = 3000.0f;
    s.mode = static_cast<uint8_t>(proto::PlayerMode::Cube);
    s.numObjects = 1;
    s.objects[0] = proto::ObjectInfo{600.0f, 9.0f, 9.0f, 18.0f, static_cast<uint8_t>(proto::ObjectKind::Hazard), 0, 0, 0};
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    uint16_t port = argc > 1 ? static_cast<uint16_t>(std::atoi(argv[1])) : proto::kDefaultPort;
    Bridge bridge;
    bridge.onLog = [](std::string_view line) { std::fprintf(stderr, "%.*s\n", static_cast<int>(line.size()), line.data()); };
    if (!bridge.listen(port)) return 1;
    std::printf("PORT %u\n", bridge.port());
    std::fflush(stdout);

    bool handshaken = false;
    uint32_t tick = 0;
    bool holding = false;

    for (;;) {
        if (!bridge.hasClient()) {
            handshaken = false;
            bridge.acceptClient(-1);
            continue;
        }
        auto msg = bridge.receive(-1);
        if (!msg) continue;

        if (auto* hello = std::get_if<proto::Hello>(&*msg)) {
            if (hello->clientVersion != 1) {  // a v1-only server: v2 clients fall back to 1
                bridge.sendError(proto::ErrorCode::VersionMismatch, "protocol version mismatch");
                bridge.dropClient();
                continue;
            }
            handshaken = bridge.sendHelloAck(proto::kDefaultTickRate);
        } else if (!handshaken) {
            bridge.sendError(proto::ErrorCode::BadMessage, "send HELLO first");
            bridge.dropClient();
        } else if (std::get_if<proto::Reset>(&*msg)) {
            tick = 0;
            holding = false;
            bridge.sendState(makeState(tick, holding));
        } else if (auto* step = std::get_if<proto::Step>(&*msg)) {
            holding = step->button != 0;
            for (uint16_t i = 0; i < step->ticks && tick < kDeathTick; ++i) ++tick;
            bridge.sendState(makeState(tick, holding));
        } else if (std::get_if<ByeMsg>(&*msg)) {
            bridge.dropClient();
        }
    }
}
