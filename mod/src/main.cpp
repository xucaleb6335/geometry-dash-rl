// Hooks: the only place the mod touches the game's control flow. Each hook
// calls the original, then tells the Controller what happened.
//
// Milestones: M0 (mod loads), M1 (find the per-tick hook), M2 (deaths, resets).
// Signatures below are from Geode's bindings for GD 2.2081; if one fails to
// compile, check $GEODE_SDK/bindings, it is part of M0.

#include <Geode/Geode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include "Controller.hpp"

using namespace geode::prelude;

class $modify(RLPlayLayer, PlayLayer) {
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) return false;
        // LEARN(M0): log the level name here so you can see the hook fire.
        gdrl::Controller::get().onLevelStart(this);
        return true;
    }

    void destroyPlayer(PlayerObject* player, GameObject* object) {
        // LEARN(M2): this is also called for things that are not real deaths.
        // Log `object` for a while and work out which calls to count, then
        // call gdrl::Controller::get().onDeath() only for those.
        PlayLayer::destroyPlayer(player, object);
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
    void processCommands(float dt, bool isHalfTick, bool isLastTick) {
        GJBaseGameLayer::processCommands(dt, isHalfTick, isLastTick);
        // LEARN(M1): confirm this runs once per physics tick (count calls per
        // second; expect the tick rate, 240 by default). If not, find the hook that does.
        gdrl::Controller::get().onTick(this);
    }
};

$on_mod(Loaded) {
    auto port = static_cast<uint16_t>(Mod::get()->getSettingValue<int64_t>("port"));
    if (!gdrl::Controller::get().start(port)) {
        log::error("GD-RL bridge failed to listen on port {}", port);
    }
}
