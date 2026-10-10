// Hooks: the only place the mod touches the game's control flow. Each hook
// calls the original, then tells the Controller what happened.
//
// Milestones: M1 (the per-tick hook), M2 (deaths, resets), M3 (lockstep).
// Signatures below are from Geode's bindings for GD 2.2081; if one fails to
// compile, check the generated headers in mod/build/bindings.

#include <Geode/Geode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include "Controller.hpp"

using namespace geode::prelude;

class $modify(RLPlayLayer, PlayLayer) {
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) return false;
        log::info("Level Opened: {}", level->m_levelName);
        gdrl::Controller::get().onLevelStart(this);
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

    void levelComplete() {
        gdrl::Controller::get().onComplete();
        PlayLayer::levelComplete();
    }

    void onQuit() {
        gdrl::Controller::get().onLevelExit();
        PlayLayer::onQuit();
    }
};

class $modify(RLBaseGameLayer, GJBaseGameLayer) {
    // Once per frame, before the frame's ticks: the safe place to reset.
    void update(float dt) {
        gdrl::Controller::get().onFrameStart(this);
        GJBaseGameLayer::update(dt);
    }

    // Runs once per physics tick (240/s, measured in M1). Note the player
    // does not move inside this call; movement happens elsewhere in the tick.
    void processCommands(float dt, bool isHalfTick, bool isLastTick) {
        GJBaseGameLayer::processCommands(dt, isHalfTick, isLastTick);
        gdrl::Controller::get().onTick(this);
    }
};

$on_mod(Loaded) {
    auto port = static_cast<uint16_t>(Mod::get()->getSettingValue<int64_t>("port"));
    if (!gdrl::Controller::get().start(port)) {
        log::error("GD-RL bridge failed to listen on port {}", port);
    }
}
