#include "Controller.hpp"

#include "StateReader.hpp"

#include <Geode/Geode.hpp>

#include <chrono>
#include <variant>

using namespace geode::prelude;

namespace gdrl {

namespace {

// How long one frame may spend serving the client before cocos draws. Keeps
// the window responsive while still running many steps per frame.
constexpr auto kFrameBudget = std::chrono::milliseconds(10);

}  // namespace

Controller& Controller::get() {
    static Controller instance;
    return instance;
}

bool Controller::start(uint16_t port) {
    m_bridge.onLog = [](std::string_view line) { log::info("{}", line); };
    return m_bridge.listen(port);
}

void Controller::onLevelStart(PlayLayer* layer) {
    m_layer = layer;
    m_lastServedFrame = m_frame;  // give its update a chance before "paused?"
    onLevelReset();
}

void Controller::onLevelExit() {
    // Every request is answered inside the frame that received it, so no
    // reply is outstanding here. Later RESET/STEPs get NOT_IN_LEVEL.
    m_layer = nullptr;
    m_holding = false;
}

void Controller::onLevelReset() {
    m_tick = 0;
    m_dead = m_completed = false;
}

void Controller::onTick(GJBaseGameLayer* layer) {
    if (layer == m_layer) ++m_ticksSeen;
}

void Controller::onDeath() {
    if (m_dead) return;  // repeat calls for the same death
    m_dead = true;
    log::info("bridge: death at tick {}", m_tick);
}

void Controller::onComplete() {
    m_completed = true;
    log::info("bridge: level completed at tick {}", m_tick);
}

bool Controller::isAgentControlling(GJBaseGameLayer* layer) const {
    return isAgentControlling() && layer == m_layer;
}

void Controller::pollIdle() {
    ++m_frame;
    if (isAgentControlling()) {
        // Normally the level's onFrame serves the client. If the level has not
        // updated for a couple of frames (pause menu), answer here so the
        // client gets an error instead of waiting for its timeout.
        if (m_frame - m_lastServedFrame <= 2) return;
        while (isAgentControlling()) {
            auto msg = m_bridge.receive(0);
            if (!msg) break;
            handle(*msg);
        }
        if (m_handshaken && !m_bridge.hasClient()) onClientLost();
        return;
    }
    if (!m_bridge.hasClient()) {
        if (m_handshaken) onClientLost();
        if (!m_bridge.acceptClient(0)) return;
    }
    // Stop once the agent takes over a level: its STEPs need onFrame.
    while (!isAgentControlling()) {
        auto msg = m_bridge.receive(0);
        if (!msg) break;
        handle(*msg);
    }
    if (m_handshaken && !m_bridge.hasClient()) onClientLost();
}

void Controller::onFrame(GJBaseGameLayer* layer, const std::function<void()>& runTick) {
    if (!isAgentControlling(layer)) return;
    m_runTick = &runTick;
    m_lastServedFrame = m_frame;

    // Serve messages until the budget is spent or the client goes quiet. While
    // Python is stepping, the next message arrives well within the wait, so
    // one frame can run thousands of ticks. While it is busy (e.g. a PPO
    // update), the frame ends after the budget and the window keeps drawing.
    auto deadline = std::chrono::steady_clock::now() + kFrameBudget;
    while (isAgentControlling()) {
        auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        if (left.count() <= 0) break;
        auto msg = m_bridge.receive(static_cast<int>(left.count()));
        if (!msg) break;
        handle(*msg);
    }
    if (m_handshaken && !m_bridge.hasClient()) onClientLost();

    m_runTick = nullptr;
}

void Controller::handle(const Incoming& msg) {
    if (auto* hello = std::get_if<proto::Hello>(&msg)) {
        if (hello->clientVersion != proto::kVersion) {
            m_bridge.sendError(proto::ErrorCode::VersionMismatch, "protocol version mismatch");
            m_bridge.dropClient();
            return;
        }
        m_handshaken = m_bridge.sendHelloAck(proto::kDefaultTickRate);
        m_lastServedFrame = m_frame;  // see pollIdle
        log::info("bridge: handshake done");
        return;
    }
    if (std::get_if<ByeMsg>(&msg)) {
        m_bridge.dropClient();
        onClientLost();
        return;
    }
    if (!m_handshaken) {
        m_bridge.sendError(proto::ErrorCode::BadMessage, "send HELLO first");
        m_bridge.dropClient();
        return;
    }
    if (!m_layer) {
        m_bridge.sendError(proto::ErrorCode::NotInLevel, "open a level first");
        return;
    }
    if (auto* reset = std::get_if<proto::Reset>(&msg)) handleReset(*reset);
    else if (auto* step = std::get_if<proto::Step>(&msg)) handleStep(*step);
}

void Controller::handleReset(const proto::Reset& msg) {
    if (msg.levelId != 0) {
        m_bridge.sendError(proto::ErrorCode::Unsupported, "only level_id 0 (current level) for now");
        return;
    }
    if (msg.startPos != -1) {
        m_bridge.sendError(proto::ErrorCode::Unsupported, "only start_pos -1 (level start) for now");
        return;
    }
    if (!m_runTick) {
        // Only reachable if the level's update is not running (e.g. paused).
        m_bridge.sendError(proto::ErrorCode::Unsupported, "level is not running (paused?)");
        return;
    }
    if (m_layer->m_isPracticeMode) {
        // resetLevel would respawn at the last checkpoint, not the start.
        m_bridge.sendError(proto::ErrorCode::Unsupported, "turn off practice mode");
        return;
    }
    // Messages are only handled at frame level (pollIdle, onFrame), never
    // inside a tick or a death hook, so resetting right here is safe.
    setButton(false);       // not still held in the new attempt
    m_layer->resetLevel();  // fires onLevelReset through the hook: tick 0
    sendCurrentState();
}

void Controller::handleStep(const proto::Step& msg) {
    // After DEAD or COMPLETED the client must RESET; repeat the terminal state.
    if (isTerminal()) {
        sendCurrentState();
        return;
    }
    if (!m_runTick) {
        // Only reachable if the level's update is not running (e.g. paused).
        m_bridge.sendError(proto::ErrorCode::Unsupported, "level is not running (paused?)");
        return;
    }
    setButton(msg.button != 0);
    // Exactly msg.ticks ticks, stopping at the tick that ends the episode.
    // ticks = 0 is a peek.
    for (uint32_t i = 0; i < msg.ticks && !isTerminal(); ++i) {
        runOneTick();
        ++m_tick;
    }
    sendCurrentState();
}

bool Controller::runOneTick() {
    m_ticksSeen = 0;
    (*m_runTick)();
    if (m_ticksSeen == 1) return true;

    // The game ran 0 or several physics ticks for one tick's worth of time.
    // That breaks "exactly N ticks"; log it (a few times) so it gets noticed.
    if (m_badTickCalls++ < 10) {
        log::warn("bridge: a one-tick update ran {} physics ticks (tick {})", m_ticksSeen, m_tick);
    }
    return false;
}

void Controller::setButton(bool down) {
    if (down == m_holding || !m_layer) return;
    // Button 1 is PlayerButton::Jump; true = player 1.
    m_layer->handleButton(down, 1, true);
    m_holding = down;
}

void Controller::onClientLost() {
    m_handshaken = false;
    if (!m_layer) return;
    // Hand the level back to the player: release the button, and restart if
    // the attempt ended (the game's own respawn was skipped under the agent).
    setButton(false);
    if (isTerminal()) m_layer->resetLevel();
}

void Controller::sendCurrentState() {
    proto::State state{};
    readState(m_layer, state);
    state.tick = m_tick;
    if (m_dead) state.flags |= proto::kDead;
    if (m_completed) state.flags |= proto::kCompleted;
    if (m_holding) state.flags |= proto::kHolding;
    m_bridge.sendState(state);
}

}  // namespace gdrl
