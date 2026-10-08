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
    m_tick = 0;
    m_dead = m_completed = false;
}

void Controller::onLevelExit() {
    m_layer = nullptr;
}

void Controller::onDeath() { m_dead = true; }
void Controller::onComplete() { m_completed = true; }

void Controller::onTick(GJBaseGameLayer* layer) {
    (void)layer;
    ++m_tick;

    // LEARN(M2): if m_resetRequested, restart the level here (start of the
    // tick, outside any death hook), clear the flags and set m_tick = 0.

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
    // LEARN(M2): request a reset (m_resetRequested = true), let onTick perform
    // it, then reply with the tick-0 state. Skip the respawn delay.
    m_bridge.sendError(proto::ErrorCode::Unsupported, "RESET not implemented yet (M2)");
}

void Controller::handleStep(const proto::Step& msg) {
    // LEARN(M2/M3):
    //   1. apply the button with m_layer->handleButton(down, 1, true) when it
    //      changes (button 1 should be jump; confirm it), track m_holding
    //   2. advance exactly msg.ticks physics ticks, stopping early on death
    //      or completion
    //   3. sendCurrentState()
    (void)msg;
    m_bridge.sendError(proto::ErrorCode::Unsupported, "STEP not implemented yet (M3)");
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
