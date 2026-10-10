#include "Controller.hpp"

#include "StateReader.hpp"

#include <Geode/Geode.hpp>

#include <variant>

using namespace geode::prelude;

namespace gdrl {

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
    onLevelReset();
}

void Controller::onLevelExit() {
    // A reply the client is still waiting for can never come now.
    if (m_resetRequested || m_stepPending) {
        m_bridge.sendError(proto::ErrorCode::NotInLevel, "level closed");
    }
    m_layer = nullptr;
    m_resetRequested = false;
    m_stepPending = false;
    m_holding = false;
}

void Controller::onLevelReset() {
    m_tick = 0;
    m_dead = m_completed = false;
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

void Controller::onFrameStart(GJBaseGameLayer* layer) {
    if (!m_resetRequested || layer != m_layer) return;
    m_resetRequested = false;

    // Release first so the button is not still held in the new attempt.
    setButton(false);
    m_layer->resetLevel();  // also fires onLevelReset through the hook
    m_stepPending = false;
    sendCurrentState();      // the tick-0 reply to RESET
}

void Controller::onTick(GJBaseGameLayer* layer) {
    if (layer == m_layer) {
        // Ticks still run on the death screen; the episode clock stops at the end.
        if (!isTerminal()) ++m_tick;
        if (m_stepPending) {
            if (m_stepTicksLeft > 0) --m_stepTicksLeft;
            if (m_stepTicksLeft == 0 || isTerminal()) finishStep();
        }
    }
    pollClient();
}

void Controller::pollClient() {
    if (!m_bridge.hasClient()) {
        m_handshaken = false;
        if (!m_bridge.acceptClient(0)) return;
    }
    // Non-blocking drain: fine for the handshake and for M2.
    // LEARN(M3): lockstep means the game must NOT advance until a STEP arrives,
    // and must stop after exactly `ticks` ticks. That changes this function:
    // block here (receive(-1)) while the agent is in control, or take over
    // tick stepping yourself. Read how GJBaseGameLayer::update turns frame
    // time into ticks (getModifiedDelta) before you decide.
    while (auto msg = m_bridge.receive(0)) handle(*msg);
}

void Controller::handle(const Incoming& msg) {
    if (auto* hello = std::get_if<proto::Hello>(&msg)) {
        if (hello->clientVersion != proto::kVersion) {
            m_bridge.sendError(proto::ErrorCode::VersionMismatch, "protocol version mismatch");
            m_bridge.dropClient();
            return;
        }
        m_handshaken = m_bridge.sendHelloAck(proto::kDefaultTickRate);
        log::info("bridge: handshake done");
        return;
    }
    if (std::get_if<ByeMsg>(&msg)) {
        m_bridge.dropClient();
        m_handshaken = false;
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
    if (m_layer->m_isPracticeMode) {
        // resetLevel would respawn at the last checkpoint, not the start.
        m_bridge.sendError(proto::ErrorCode::Unsupported, "turn off practice mode");
        return;
    }
    // Never reset from here: we may be inside the game's tick. onFrameStart
    // does it before the next frame and sends the tick-0 STATE.
    m_resetRequested = true;
}

void Controller::handleStep(const proto::Step& msg) {
    // After DEAD or COMPLETED the client must RESET; repeat the terminal state.
    if (isTerminal()) {
        sendCurrentState();
        return;
    }
    setButton(msg.button != 0);
    if (msg.ticks == 0) {
        sendCurrentState();  // a peek
        return;
    }
    // onTick counts the ticks down and replies. LEARN(M3): this does not stop
    // the game between steps yet; lockstep must make it wait for the next STEP.
    m_stepPending = true;
    m_stepTicksLeft = msg.ticks;
}

void Controller::setButton(bool down) {
    if (down == m_holding || !m_layer) return;
    // Button 1 is PlayerButton::Jump; true = player 1.
    m_layer->handleButton(down, 1, true);
    m_holding = down;
}

void Controller::finishStep() {
    m_stepPending = false;
    sendCurrentState();
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
