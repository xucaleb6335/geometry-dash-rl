// Hooks: the only place the mod touches the game's control flow. Each hook
// calls the original, then tells the Controller what happened.
//
// Milestones: M1 (the per-tick hook), M2 (deaths, resets), M3 (lockstep).
// Signatures below are from Geode's bindings for GD 2.2081; if one fails to
// compile, check the generated headers in mod/build/bindings.

#include <Geode/Geode.hpp>
#include <Geode/modify/CCScheduler.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include "Controller.hpp"
#include "Overlay.hpp"

using namespace geode::prelude;

class $modify(RLPlayLayer, PlayLayer) {
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) return false;
        log::info("Level Opened: {}", level->m_levelName);
        gdrl::Controller::get().onLevelStart(this);
        gdrl::Overlay::get().onLevelStart(this);
        return true;
    }

    void destroyPlayer(PlayerObject* player, GameObject* object) {
        // When a level starts, GD kills the player with an invisible
        // "anticheat spike" to check collisions work. That is not a death.
        // (It fires about 13 times per attempt.) GD can also call this several
        // times for one death; onDeath only counts the first.
        bool real = object != m_anticheatSpike;
        PlayLayer::destroyPlayer(player, object);
        if (real) gdrl::Controller::get().onDeath();
    }

    // GD's own restart after a death (about a second later). The agent
    // resets through RESET instead, so skip it while a client is in control.
    void delayedResetLevel() {
        if (gdrl::Controller::get().isAgentControlling()) return;
        PlayLayer::delayedResetLevel();
    }

    void resetLevel() {
        PlayLayer::resetLevel();
        gdrl::Controller::get().onLevelReset();
        gdrl::Overlay::get().onReset();
    }

    // GD pauses when its window loses focus (e.g. clicking the terminal that
    // runs the Python client), which would stall every reply. Not under the agent.
    void pauseGame(bool unfocused) {
        if (unfocused && gdrl::Controller::get().isAgentControlling()) return;
        PlayLayer::pauseGame(unfocused);
    }

    // The results screen after completing a level; skipped under the agent.
    void showEndLayer() {
        if (gdrl::Controller::get().isAgentControlling()) return;
        PlayLayer::showEndLayer();
    }

    // The player crossed the end; runs inside a physics tick, so under the
    // agent the STEP that crossed it already reports COMPLETED.
    void playEndAnimationToPos(CCPoint position) {
        PlayLayer::playEndAnimationToPos(position);
        gdrl::Controller::get().onEndAnimation();
    }

    // GD calls this when the end animation finishes, in real time. Under
    // lockstep the client has usually reset by then: that call is stale and
    // is dropped (it would mark the next attempt completed and, once the
    // client has gone, open the end screen over a fresh attempt).
    void levelComplete() {
        auto& controller = gdrl::Controller::get();
        if (!controller.completionIsCurrent()) return;
        controller.onComplete();
        PlayLayer::levelComplete();
    }

    void onQuit() {
        gdrl::Controller::get().onLevelExit();
        gdrl::Overlay::get().onLevelExit();
        PlayLayer::onQuit();
    }
};

class $modify(RLBaseGameLayer, GJBaseGameLayer) {
    // One frame of the level. Normally the game turns the frame's real time
    // into ticks. Under the agent (M3 lockstep) the frame runs no ticks of its
    // own: the Controller serves the client and each STEP runs its ticks
    // through runOneTick. No STEP, no ticks.
    void update(float dt) {
        auto& controller = gdrl::Controller::get();
        if (!controller.isAgentControlling(this)) GJBaseGameLayer::update(dt);
        else controller.onFrame(this, [this] { runOneTick(); });
        // Setting show-hitboxes: redraw the agent's view for this frame.
        if (auto* level = typeinfo_cast<PlayLayer*>(this)) gdrl::Overlay::get().onFrame(level);
    }

    // Exactly one physics tick: feed the game one tick's worth of time with an
    // empty time bank (m_extraDelta), so nothing left over from earlier adds or
    // drops a tick, and no resume delay (GD skips ticks for a few frames after
    // a reset or unpause). float(1/240) is a hair above 1/240, so the game's
    // tick count for it is 1 whether it rounds or floors. The Controller checks
    // that exactly one tick ran.
    void runOneTick() {
        m_extraDelta = 0.0;
        m_resumeTimer = 0;
        GJBaseGameLayer::update(1.0f / gdrl::proto::kDefaultTickRate);
    }

    // Runs once per physics tick (240/s, measured in M1). Note the player
    // does not move inside this call; movement happens elsewhere in the tick.
    void processCommands(float dt, bool isHalfTick, bool isLastTick) {
        GJBaseGameLayer::processCommands(dt, isHalfTick, isLastTick);
        gdrl::Controller::get().onTick(this);
    }
};

// Once per frame for the whole game, menus included: accept clients and
// answer them while no level is under the agent (handshake, NOT_IN_LEVEL).
class $modify(RLScheduler, CCScheduler) {
    void update(float dt) {
        gdrl::Controller::get().pollIdle();
        CCScheduler::update(dt);
    }
};

$on_mod(Loaded) {
    auto port = static_cast<uint16_t>(Mod::get()->getSettingValue<int64_t>("port"));
    if (!gdrl::Controller::get().start(port)) {
        log::error("GD-RL bridge failed to listen on port {}", port);
    }
}
