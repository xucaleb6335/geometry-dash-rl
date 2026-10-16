// Glue between the hooks and the bridge: owns the socket, the episode state and
// the lockstep loop. One instance, main thread only.
//
// Lockstep (M3): while a client is connected and a level is open, the level
// runs no ticks on its own. Each frame, onFrame serves the client's messages
// for a few milliseconds; a STEP of N ticks runs exactly N one-tick updates
// right there, then replies. No STEP, no ticks: the game waits for Python.
#pragma once

#include "Bridge.hpp"
#include "Protocol.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

class PlayLayer;
class GJBaseGameLayer;
class StartPosObject;

namespace gdrl {

struct SavedState;  // a v2 snapshot (Controller.cpp)

class Controller {
public:
    static Controller& get();

    bool start(uint16_t port);

    // Called from the hooks in main.cpp.
    void onLevelStart(PlayLayer* layer);
    void onLevelExit();
    void onLevelReset();                   // after any resetLevel, ours or the game's
    void onTick(GJBaseGameLayer* layer);   // once per physics tick (confirmed in M1)
    void onDeath();                        // a real death, as decided in M2
    void onComplete();
    // The end animation started (the player crossed the level end), inside a
    // physics tick. That is the completion under the agent: GD's own
    // levelComplete() comes later from a real-time animation, after the client
    // may already have reset.
    void onEndAnimation();
    // False if levelComplete() belongs to an attempt that has since been reset
    // (a stale callback from the end animation); the hook then skips it, so it
    // can neither mark the new attempt completed nor open the end screen.
    bool completionIsCurrent() const;

    // Every frame, before cocos runs its scheduled updates. Accepts clients and
    // answers messages while no level is under the agent (handshake, errors),
    // or while that level is not updating (pause menu: requests get errors).
    void pollIdle();

    // The level's frame while the agent is in control (instead of the game's
    // own update). `runTick` must advance the game by exactly one tick.
    void onFrame(GJBaseGameLayer* layer, const std::function<void()>& runTick);

    bool isAgentControlling() const { return m_handshaken && m_layer != nullptr; }
    // Same, and `layer` is the level the agent plays (not e.g. the editor).
    bool isAgentControlling(GJBaseGameLayer* layer) const;

private:
    Controller();
    ~Controller();

    void handle(const Incoming& msg);
    void handleReset(const proto::Reset& msg);
    void handleStep(const proto::Step& msg);
    void handleSnapshot(const SnapshotMsg& msg);
    void handleSave(int slot);
    void handleLoad(int slot);
    bool runOneTick();
    void setButton(bool down);
    void onClientLost();
    void removeEndScreen();
    void sendCurrentState();
    bool isTerminal() const { return m_dead || m_completed; }

    Bridge m_bridge;
    PlayLayer* m_layer = nullptr;
    StartPosObject* m_defaultStartPos = nullptr;  // the level's own start pos (RESET -1)
    bool m_defaultStartPosKnown = false;
    bool m_handshaken = false;

    // Episode bookkeeping, written by the hooks, read when building STATE.
    uint32_t m_tick = 0;  // ticks run by STEPs since the last reset
    bool m_dead = false;
    bool m_completed = false;
    bool m_holding = false;
    uint64_t m_attempt = 0;       // counts resets
    int64_t m_endAttempt = -1;    // attempt whose end animation started (-1: none seen)

    // Set only while onFrame runs: how to advance one tick, and how many
    // physics ticks (processCommands calls) the last call actually ran.
    const std::function<void()>* m_runTick = nullptr;
    int m_ticksSeen = 0;
    int m_badTickCalls = 0;

    // Frame counter (pollIdle) and the last frame the level's update served
    // the client, to notice a level that stopped updating (pause menu).
    uint64_t m_frame = 0;
    uint64_t m_lastServedFrame = 0;

    // v2 snapshots, one per slot (empty = nullptr). Cleared when the level exits.
    std::vector<std::unique_ptr<SavedState>> m_snapshots;
};

}  // namespace gdrl
