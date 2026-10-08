// Glue between the hooks and the bridge: owns the socket, the episode state and
// (eventually) the lockstep loop. One instance, main thread only.
#pragma once

#include "Bridge.hpp"
#include "Protocol.hpp"

#include <cstdint>

class PlayLayer;
class GJBaseGameLayer;

namespace gdrl {

class Controller {
public:
    static Controller& get();

    bool start(uint16_t port);

    // Called from the hooks in main.cpp.
    void onLevelStart(PlayLayer* layer);
    void onLevelExit();
    void onTick(GJBaseGameLayer* layer);  // once per physics tick (confirm in M1)
    void onDeath();                        // a real death, as decided in M2
    void onComplete();

    bool isAgentControlling() const { return m_handshaken && m_layer != nullptr; }

private:
    Controller() = default;

    void pollClient();
    void handle(const Incoming& msg);
    void handleReset(const proto::Reset& msg);
    void handleStep(const proto::Step& msg);
    void sendCurrentState();

    Bridge m_bridge;
    PlayLayer* m_layer = nullptr;
    bool m_handshaken = false;

    // Episode bookkeeping, written by the hooks, read when building STATE.
    uint32_t m_tick = 0;
    bool m_dead = false;
    bool m_completed = false;
    bool m_holding = false;
    bool m_resetRequested = false;  // reset at the start of the next tick, never inside a hook
};

}  // namespace gdrl
