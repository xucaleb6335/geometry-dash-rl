#include "Controller.hpp"

#include "StateReader.hpp"

#include <Geode/Geode.hpp>

#include <algorithm>
#include <chrono>
#include <variant>
#include <vector>

using namespace geode::prelude;

namespace gdrl {

namespace {

// How long one frame may spend serving the client before cocos draws. Keeps
// the window responsive (~10 redraws/s) while running many steps per frame.
// Was 10 ms: on Bloodbath a frame draw takes about as long, so half of the
// time went to drawing (profile: the client waited 1.9 ms per 1.3 ms STEP).
constexpr auto kFrameBudget = std::chrono::milliseconds(100);

// PlayerObject fields a practice checkpoint (PlayerCheckpoint) does not keep,
// or keeps rounded, that change physics or input handling. Copied verbatim on
// SAVE and written back after LOAD. If the in-game determinism check
// (gdrl.tools.snapshot_check) finds a mismatch, the missing field goes here.
#define GDRL_PLAYER_FIELDS(X)                                                            \
    X(m_yVelocity) X(m_fallSpeed) X(m_isOnGround) X(m_isOnGround2) X(m_isOnGround3)      \
    X(m_isOnSlope) X(m_wasOnSlope) X(m_slopeVelocity) X(m_isDashing) X(m_vehicleSize)    \
    X(m_playerSpeed) X(m_isUpsideDown) X(m_isShip) X(m_isBird) X(m_isBall) X(m_isDart)   \
    X(m_isRobot) X(m_isSpider) X(m_isSwing) X(m_isSideways) X(m_isGoingLeft)            \
    X(m_jumpBuffered) X(m_stateJumpBuffered) X(m_wasJumpBuffered) X(m_stateRingJump)     \
    X(m_stateRingJump2) X(m_touchedRing) X(m_touchedCustomRing) X(m_touchedGravityPortal) \
    X(m_touchedPad) X(m_ringJumpRelated) X(m_isAccelerating) X(m_lastLandTime)          \
    X(m_lastJumpTime) X(m_lastFlipTime) X(m_totalTime) X(m_gravity) X(m_speedMultiplier) \
    X(m_yStart) X(m_accelerationOrSpeed) X(m_snapDistance) X(m_collidedTopMinY)         \
    X(m_collidedBottomMaxY) X(m_collidedLeftMaxX) X(m_collidedRightMinX)                \
    X(m_lastGroundedPos) X(m_platformerVelocityRelated) X(m_maybeIsBoosted)             \
    X(m_touchedRings) X(m_ringRelatedSet) X(m_holdingButtons) X(m_holdingLeft)          \
    X(m_holdingRight) X(m_hasEverJumped) X(m_hasEverHitRing)

struct PlayerFix {
#define GDRL_DECLARE(f) decltype(PlayerObject::f) f{};
    GDRL_PLAYER_FIELDS(GDRL_DECLARE)
#undef GDRL_DECLARE
    float rotation = 0.0f;
    cocos2d::CCPoint position;
    bool valid = false;

    void save(PlayerObject* p) {
        if (!p) return;
#define GDRL_SAVE(f) f = p->f;
        GDRL_PLAYER_FIELDS(GDRL_SAVE)
#undef GDRL_SAVE
        rotation = p->getRotation();
        position = p->getPosition();
        valid = true;
    }

    void restore(PlayerObject* p) const {
        if (!p || !valid) return;
#define GDRL_RESTORE(f) p->f = f;
        GDRL_PLAYER_FIELDS(GDRL_RESTORE)
#undef GDRL_RESTORE
        p->setRotation(rotation);
        p->setPosition(position);
    }
};

}  // namespace

// One SAVE_STATE: GD's own practice checkpoint (game state, triggers, object
// states) plus the player fields it misses and our episode bookkeeping.
struct SavedState {
    CheckpointObject* checkpoint = nullptr;  // retained
    PlayerFix p1, p2;
    uint32_t tick = 0;
    bool holding = false;

    ~SavedState() {
        if (checkpoint) checkpoint->release();
    }
};

Controller::Controller() = default;
Controller::~Controller() = default;

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
    m_defaultStartPosKnown = false;  // read on the first RESET, once the level has finished loading
    m_lastServedFrame = m_frame;  // give its update a chance before "paused?"
    onLevelReset();
}

void Controller::onLevelExit() {
    // Every request is answered inside the frame that received it, so no
    // reply is outstanding here. Later RESET/STEPs get NOT_IN_LEVEL.
    m_layer = nullptr;
    m_holding = false;
    m_snapshots.clear();  // checkpoints belong to the level being left
}

void Controller::onLevelReset() {
    ++m_attempt;
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
    if (m_completed) return;
    m_completed = true;
    log::info("bridge: level completed at tick {}", m_tick);
}

void Controller::onEndAnimation() {
    if (m_endAttempt == static_cast<int64_t>(m_attempt)) return;
    m_endAttempt = static_cast<int64_t>(m_attempt);
    if (isAgentControlling()) onComplete();
}

bool Controller::completionIsCurrent() const {
    return m_endAttempt < 0 || m_endAttempt == static_cast<int64_t>(m_attempt);
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
    // Setting watch-mode: at most one STEP per frame, so a client pacing its
    // STEPs in real time (60/s of 4 ticks) gets one rendered frame per STEP
    // instead of several STEPs batched into each ~100 ms frame (~10 fps).
    const bool watch = Mod::get()->getSettingValue<bool>("watch-mode");
    auto deadline = std::chrono::steady_clock::now() + kFrameBudget;
    while (isAgentControlling()) {
        auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now());
        if (left.count() <= 0) break;
        auto msg = m_bridge.receive(static_cast<int>(left.count()));
        if (!msg) break;
        handle(*msg);
        if (watch && std::holds_alternative<proto::Step>(*msg)) break;
    }
    if (m_handshaken && !m_bridge.hasClient()) onClientLost();

    m_runTick = nullptr;
}

void Controller::handle(const Incoming& msg) {
    if (auto* hello = std::get_if<proto::Hello>(&msg)) {
        if (hello->clientVersion < proto::kMinVersion || hello->clientVersion > proto::kVersion) {
            m_bridge.sendError(proto::ErrorCode::VersionMismatch, "protocol version mismatch");
            m_bridge.dropClient();
            return;
        }
        m_bridge.setVersion(hello->clientVersion);
        m_handshaken = m_bridge.sendHelloAck(proto::kDefaultTickRate, proto::kMaxSnapshots);
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
    else if (auto* snap = std::get_if<SnapshotMsg>(&msg)) handleSnapshot(*snap);
}

void Controller::handleReset(const proto::Reset& msg) {
    if (msg.levelId != 0) {
        m_bridge.sendError(proto::ErrorCode::Unsupported, "only level_id 0 (current level) for now");
        return;
    }
    // -1: where GD itself starts this level (its active Start Pos, if the level
    // has one; v1 behaviour). -2: the real level start, ignoring Start Pos
    // objects. k >= 0: the k-th Start Pos object, ordered by x.
    if (!m_defaultStartPosKnown) {
        m_defaultStartPos = m_layer->m_startPosObject;  // where GD itself starts this level
        m_defaultStartPosKnown = true;
    }
    StartPosObject* startPos = msg.startPos == -1 ? m_defaultStartPos : nullptr;
    if (msg.startPos >= 0) {
        std::vector<GameObject*> starts;
        for (auto* obj : CCArrayExt<GameObject*>(m_layer->m_objects)) {
            if (obj && obj->m_objectID == 31) starts.push_back(obj);
        }
        std::sort(starts.begin(), starts.end(), [](GameObject* a, GameObject* b) {
            return a->getPositionX() < b->getPositionX();
        });
        if (msg.startPos >= static_cast<int32_t>(starts.size())) {
            m_bridge.sendError(proto::ErrorCode::Unsupported, "no such Start Pos in this level");
            return;
        }
        startPos = static_cast<StartPosObject*>(starts[msg.startPos]);
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
    removeEndScreen();
    // GD caches the spawn state of the active Start Pos as m_currentCheckpoint
    // and respawns from it; drop it so the new Start Pos is used (same as the
    // common Start Pos switcher mods).
    // Player rotation state survives resetLevel (rotation actions and the
    // smoothing fields), so it depended on how the last attempt ended.
    for (auto* p : {m_layer->m_player1, m_layer->m_player2}) {
        if (!p) continue;
        p->stopAllActions();
        p->m_rotationSpeed = 0.f;
        p->m_rotateSpeed = 0.f;
        p->m_isRotating = false;
        p->m_isBallRotating = false;
        p->m_isBallRotating2 = false;
        p->m_shipRotation = CCPointZero;
        p->setRotation(0.f);
    }
    m_layer->m_currentCheckpoint = nullptr;
    m_layer->setStartPosObject(startPos);
    m_layer->resetLevel();  // fires onLevelReset through the hook: tick 0
    // Reset twice. A single resetLevel leaves some state from how the last
    // attempt ended (player rotation state, trigger-driven object state): on
    // Bloodbath consecutive attempts alternated between two dynamics (same
    // inputs, y off by 0.4 at x 10,649; the greedy policy died at 11,819 or
    // 10,772 by parity). The second reset starts from a fresh tick-0 attempt.
    m_layer->m_currentCheckpoint = nullptr;
    m_layer->setStartPosObject(startPos);
    m_layer->resetLevel();
    sendCurrentState();
}

void Controller::handleSnapshot(const SnapshotMsg& msg) {
    if (m_bridge.version() < 2) {
        m_bridge.sendError(proto::ErrorCode::BadMessage, "snapshots need protocol version 2");
        return;
    }
    if (msg.msg.slot < 0 || msg.msg.slot >= static_cast<int32_t>(proto::kMaxSnapshots)) {
        m_bridge.sendError(proto::ErrorCode::NoSnapshot, "slot out of range");
        return;
    }
    if (m_snapshots.size() < proto::kMaxSnapshots) m_snapshots.resize(proto::kMaxSnapshots);
    if (!m_runTick) {
        m_bridge.sendError(proto::ErrorCode::Unsupported, "level is not running (paused?)");
        return;
    }
    switch (msg.type) {
        case proto::MsgType::SaveState: handleSave(msg.msg.slot); break;
        case proto::MsgType::LoadState: handleLoad(msg.msg.slot); break;
        default:  // DropState
            m_snapshots[msg.msg.slot].reset();
            sendCurrentState();
    }
}

void Controller::handleSave(int slot) {
    if (isTerminal()) {
        m_bridge.sendError(proto::ErrorCode::Unsupported, "cannot save a dead or completed state");
        return;
    }
    auto saved = std::make_unique<SavedState>();
    saved->checkpoint = m_layer->createCheckpoint();
    if (!saved->checkpoint) {
        m_bridge.sendError(proto::ErrorCode::Unsupported, "createCheckpoint failed");
        return;
    }
    saved->checkpoint->retain();
    saved->p1.save(m_layer->m_player1);
    saved->p2.save(m_layer->m_player2);
    saved->tick = m_tick;
    saved->holding = m_holding;
    m_snapshots[slot] = std::move(saved);
    sendCurrentState();
}

void Controller::handleLoad(int slot) {
    auto* saved = m_snapshots[slot].get();
    if (!saved) {
        m_bridge.sendError(proto::ErrorCode::NoSnapshot, "slot is empty");
        return;
    }
    // Mod setting snapshot-load: 0 = GD's practice-mode respawn with our
    // checkpoint as the last one (resets the level and loads the checkpoint:
    // objects, triggers, player); 1 = resetLevel, then loadFromCheckpoint
    // directly (a fallback if 0 misbehaves).
    auto method = Mod::get()->getSettingValue<int64_t>("snapshot-load");
    if (method == 0) {
        bool practice = m_layer->m_isPracticeMode;
        m_layer->m_isPracticeMode = true;
        m_layer->m_checkpointArray->addObject(saved->checkpoint);
        m_layer->resetLevel();
        m_layer->m_checkpointArray->removeObject(saved->checkpoint, false);
        m_layer->m_isPracticeMode = practice;
    } else {
        m_layer->resetLevel();
        m_layer->loadFromCheckpoint(saved->checkpoint);
    }
    saved->p1.restore(m_layer->m_player1);
    saved->p2.restore(m_layer->m_player2);
    // Our bookkeeping: the attempt continues from the saved tick, and the
    // button is as it was, without a press or release event reaching GD.
    m_tick = saved->tick;
    m_dead = m_completed = false;
    m_holding = saved->holding;
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
        // Backup for the playEndAnimationToPos hook: the flag GD sets when
        // the player crosses the end.
        if (m_layer && m_layer->m_levelEndAnimationStarted) onEndAnimation();
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
    // the attempt died (the game's own respawn was skipped under the agent).
    // A completed attempt is left alone: GD finishes its end animation and
    // shows the normal end screen, from which the level can be left (or the
    // next client's RESET removes it). Resetting here instead let the end
    // animation finish over a fresh attempt and open the end screen on it.
    setButton(false);
    if (m_dead) m_layer->resetLevel();
}

void Controller::removeEndScreen() {
    // The results screen (EndLevelLayer, a drop-down layer on the scene) can
    // be up if a level was beaten while no client was connected.
    std::vector<CCNode*> found;
    for (CCNode* parent : {static_cast<CCNode*>(CCDirector::get()->getRunningScene()),
                           static_cast<CCNode*>(m_layer)}) {
        if (!parent) continue;
        for (auto* child : CCArrayExt<CCNode*>(parent->getChildren())) {
            if (typeinfo_cast<EndLevelLayer*>(child)) found.push_back(child);
        }
    }
    for (auto* node : found) {
        node->removeFromParentAndCleanup(true);
        log::info("bridge: removed the end screen");
    }
}

void Controller::sendCurrentState() {
    proto::StateV2 state{};
    readState(m_layer, state);
    state.tick = m_tick;
    if (m_dead) state.flags |= proto::kDead;
    if (m_completed) state.flags |= proto::kCompleted;
    if (m_holding) state.flags |= proto::kHolding;
    m_bridge.sendState(state);
}

}  // namespace gdrl
