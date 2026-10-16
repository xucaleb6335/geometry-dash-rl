// GD Agent Control: lets an AI agent drive the game over localhost TCP
// (JSON lines, port 5556 by default). See ../PROTOCOL.md.
//
// The socket runs on its own threads (LineServer). Each request is queued on
// the main thread with queueInMainThread; the client thread waits for the
// reply (or a timeout) and sends it back.

#include <Geode/Geode.hpp>
#include <Geode/modify/CCScheduler.hpp>
#include <Geode/modify/MenuLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include "Commands.hpp"
#include "LineServer.hpp"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

using namespace geode::prelude;

namespace {

struct Pending {
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    std::string reply;
};

std::string errorLine(matjson::Value const& id, std::string const& cmd, std::string const& code,
                      std::string const& message) {
    auto out = matjson::Value::object();
    out["ok"] = false;
    if (!id.isNull()) out["seq"] = id;
    if (!cmd.empty()) out["cmd"] = cmd;
    out["code"] = code;
    out["error"] = message;
    return out.dump(matjson::NO_INDENTATION);
}

// Runs on a client thread.
std::string handleLine(std::string const& line) {
    auto parsed = matjson::parse(line);
    if (!parsed) return errorLine(nullptr, "", "bad_json", "request is not valid JSON");
    auto req = parsed.unwrap();
    if (!req.isObject()) return errorLine(nullptr, "", "bad_request", "request must be a JSON object");
    matjson::Value id = req.contains("seq") ? req["seq"] : matjson::Value(nullptr);
    std::string cmd;
    if (req.contains("cmd")) {
        if (auto c = req["cmd"].asString()) cmd = c.unwrap();
    }

    // Answered on the network thread: proves the socket is alive even when the
    // main thread is stuck (gdctl status uses it to tell "hung" from "closed").
    if (cmd == "net_ping") {
        auto out = matjson::Value::object();
        out["ok"] = true;
        if (!id.isNull()) out["seq"] = id;
        out["cmd"] = cmd;
        out["thread"] = "net";
        out["frame"] = static_cast<int64_t>(gdctl::state().frame);
        return out.dump(matjson::NO_INDENTATION);
    }

    double timeout = 30.0;
    if (req.contains("timeout")) {
        if (auto t = req["timeout"].asDouble()) timeout = t.unwrap();
        else if (auto ti = req["timeout"].asInt()) timeout = static_cast<double>(ti.unwrap());
    }
    timeout = std::clamp(timeout, 0.5, 600.0);

    auto pending = std::make_shared<Pending>();
    queueInMainThread([req, id, cmd, pending] {
        if (Mod::get()->getSettingValue<bool>("log-commands") && cmd != "ping" && cmd != "scene") {
            log::info("control: {}", cmd);
        }
        gdctl::dispatch(req, [id, cmd, pending](matjson::Value out) {
            if (!out.isObject()) out = matjson::Value::object();
            if (!out.contains("ok")) out["ok"] = true;
            if (!id.isNull()) out["seq"] = id;
            out["cmd"] = cmd;
            auto text = out.dump(matjson::NO_INDENTATION);
            std::lock_guard lock(pending->mutex);
            if (pending->done) return;
            pending->reply = std::move(text);
            pending->done = true;
            pending->cv.notify_all();
        });
    });

    std::unique_lock lock(pending->mutex);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(static_cast<int64_t>(timeout * 1000));
    if (!pending->cv.wait_until(lock, deadline, [&] { return pending->done; })) {
        pending->done = true;  // a late reply is dropped
        return errorLine(id, cmd, "timeout",
                         "main thread did not answer in " + std::to_string(static_cast<int>(timeout)) +
                             " s (game busy, minimized or hung?)");
    }
    return pending->reply;
}

}  // namespace

class $modify(CtlPlayLayer, PlayLayer) {
    // GD pauses when its window loses focus. With prevent_pause on, it never
    // does, with or without an RL client (the bridge only covers the latter).
    void pauseGame(bool unfocused) {
        if (unfocused && gdctl::preventPause()) return;
        PlayLayer::pauseGame(unfocused);
    }
};

class $modify(CtlMenuLayer, MenuLayer) {
    bool init() {
        if (!MenuLayer::init()) return false;
        if (!gdctl::state().menuReached) log::info("control: main menu reached");
        gdctl::state().menuReached = true;
        return true;
    }
};

class $modify(CtlScheduler, CCScheduler) {
    void update(float dt) {
        ++gdctl::state().frame;
        CCScheduler::update(dt);
    }
};

$on_mod(Loaded) {
    auto port = static_cast<uint16_t>(Mod::get()->getSettingValue<int64_t>("port"));
    gdctl::state().port = port;
    // Leaked on purpose: its threads run until the process exits.
    auto* server = new gdctl::LineServer();
    bool ok = server->start(port, handleLine, [](std::string_view line) { log::info("{}", line); });
    if (!ok) log::error("GD agent control failed to listen on port {}", port);
}
