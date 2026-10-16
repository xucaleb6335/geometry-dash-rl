// Command dispatch for the agent-control protocol (see ../PROTOCOL.md).
// Everything here runs on the game's main thread.
#pragma once

#include <Geode/Geode.hpp>

#include <functional>
#include <optional>
#include <string>

namespace gdctl {

// Called exactly once with the reply object ("ok", "id" and "cmd" are added
// by the caller if missing). May be called later, from a later frame.
using Reply = std::function<void(matjson::Value)>;

// Run one request. Must be called on the main thread.
void dispatch(matjson::Value const& req, Reply reply);

// Shared state, read by the hooks in main.cpp.
struct State {
    std::optional<bool> preventPauseOverride;  // runtime value of prevent_pause
    bool menuReached = false;                  // MenuLayer::init ran at least once
    uint64_t frame = 0;                        // CCScheduler frames since load
    uint16_t port = 0;
};
State& state();
bool preventPause();  // effective value (override, else the mod setting)

}  // namespace gdctl
