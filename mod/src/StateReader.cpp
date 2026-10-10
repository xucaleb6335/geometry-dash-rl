#include "StateReader.hpp"

#include <Geode/Geode.hpp>

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

using namespace geode::prelude;

namespace gdrl {

namespace {

constexpr float kBlock = 30.0f;          // game units per block
constexpr float kLookAhead = 20 * kBlock;  // how far ahead of the player to search
constexpr float kLookVertical = 20 * kBlock;  // how far above and below the player
constexpr float kGroundY = 90.0f;  // GD's ground surface; the protocol puts it at y = 0
constexpr float kTicksPerVelUnit = 60.0f;  // m_yVelocity is units per 1/60 s
// Measured in game (2026-10-10): every mode except the wave moves 0.9 x
// m_yVelocity * 60 per second; the wave moves exactly m_yVelocity * 60.
constexpr float kNonWaveVelScale = 0.9f;

// m_playerSpeed is a multiplier, not a speed. Units/s for each speed portal;
// 0.9, 1.1 and 1.3 were checked against the x position in the 1 Hz log.
float xSpeedUnitsPerSecond(float multiplier) {
    struct Speed {
        float multiplier;
        float unitsPerSecond;
    };
    constexpr Speed kSpeeds[] = {
        {0.7f, 251.16f},  // slow
        {0.9f, 311.58f},  // normal
        {1.1f, 387.42f},  // 2x
        {1.3f, 468.00f},  // 3x
        {1.6f, 576.00f},  // 4x
    };
    const Speed* best = &kSpeeds[0];
    for (const auto& speed : kSpeeds) {
        if (std::abs(speed.multiplier - multiplier) < std::abs(best->multiplier - multiplier)) {
            best = &speed;
        }
    }
    return best->unitsPerSecond;
}

// Map GD's object type to the protocol's kind. nullopt = never send
// (decorations, triggers, coins).
std::optional<proto::ObjectKind> classify(GameObjectType type) {
    switch (type) {
        case GameObjectType::Solid:
        case GameObjectType::Breakable:
        case GameObjectType::Slope:
            return proto::ObjectKind::Solid;
        case GameObjectType::Hazard:
        case GameObjectType::AnimatedHazard:
            return proto::ObjectKind::Hazard;
        case GameObjectType::YellowJumpRing:
        case GameObjectType::PinkJumpRing:
        case GameObjectType::GravityRing:
        case GameObjectType::GreenRing:
        case GameObjectType::DropRing:
        case GameObjectType::RedJumpRing:
        case GameObjectType::CustomRing:
        case GameObjectType::DashRing:
        case GameObjectType::GravityDashRing:
        case GameObjectType::SpiderOrb:
        case GameObjectType::TeleportOrb:
            return proto::ObjectKind::Orb;
        case GameObjectType::YellowJumpPad:
        case GameObjectType::PinkJumpPad:
        case GameObjectType::GravityPad:
        case GameObjectType::RedJumpPad:
        case GameObjectType::SpiderPad:
            return proto::ObjectKind::Pad;
        case GameObjectType::InverseGravityPortal:
        case GameObjectType::NormalGravityPortal:
        case GameObjectType::ShipPortal:
        case GameObjectType::CubePortal:
        case GameObjectType::InverseMirrorPortal:
        case GameObjectType::NormalMirrorPortal:
        case GameObjectType::BallPortal:
        case GameObjectType::RegularSizePortal:
        case GameObjectType::MiniSizePortal:
        case GameObjectType::UfoPortal:
        case GameObjectType::DualPortal:
        case GameObjectType::SoloPortal:
        case GameObjectType::WavePortal:
        case GameObjectType::RobotPortal:
        case GameObjectType::TeleportPortal:
        case GameObjectType::SpiderPortal:
        case GameObjectType::SwingPortal:
        case GameObjectType::GravityTogglePortal:
            return proto::ObjectKind::Portal;
        case GameObjectType::Decoration:
        case GameObjectType::Modifier:
        case GameObjectType::SecretCoin:
        case GameObjectType::UserCoin:
        case GameObjectType::Collectible:
        case GameObjectType::EnterEffectObject:
            return std::nullopt;
        default:
            return proto::ObjectKind::Other;
    }
}

}  // namespace

void readState(PlayLayer* layer, proto::State& out) {
    out = proto::State{};
    if (!layer) return;

    auto* player = layer->m_player1;
    if (!player) return;

    // Game coordinates: GD's ground is at y = kGroundY. Everything sent is
    // shifted so the ground is at 0 (PROTOCOL.md).
    float gameX = player->getPositionX();
    float gameY = player->getPositionY();
    out.playerX = gameX;
    out.playerY = gameY - kGroundY;

    // m_yVelocity is already world direction (+ = up), upside down included.
    float velY = static_cast<float>(player->m_yVelocity) * kTicksPerVelUnit;
    out.velY = player->m_isDart ? velY : velY * kNonWaveVelScale;

    out.xSpeed = xSpeedUnitsPerSecond(player->m_playerSpeed);
    out.levelLength = layer->m_levelLength;

    // cocos rotation is clockwise degrees and never wraps; keep it in [0, 360)
    // like the mock game.
    float rotation = std::fmod(player->getRotation(), 360.0f);
    out.rotation = rotation < 0 ? rotation + 360.0f : rotation;

    if (player->m_isOnGround) out.flags |= proto::kOnGround;
    if (player->m_isUpsideDown) out.flags |= proto::kUpsideDown;

    // GD has no cube flag: cube is what's left when no other mode flag is set.
    // m_isBird is the UFO and m_isDart is the wave.
    proto::PlayerMode mode;
    if (player->m_isShip) mode = proto::PlayerMode::Ship;
    else if (player->m_isBall) mode = proto::PlayerMode::Ball;
    else if (player->m_isBird) mode = proto::PlayerMode::Ufo;
    else if (player->m_isDart) mode = proto::PlayerMode::Wave;
    else if (player->m_isRobot) mode = proto::PlayerMode::Robot;
    else if (player->m_isSpider) mode = proto::PlayerMode::Spider;
    else if (player->m_isSwing) mode = proto::PlayerMode::Swing;
    else mode = proto::PlayerMode::Cube;
    out.mode = static_cast<uint8_t>(mode);

    // Objects: search a box from one block behind the player's left edge to
    // kLookAhead in front, kLookVertical above and below. CCRect is
    // (bottom-left x, bottom-left y, width, height).
    float playerLeft = player->getObjectRect().getMinX();
    float searchLeft = playerLeft - kBlock;
    CCRect search(searchLeft, gameY - kLookVertical,
                  gameX + kLookAhead - searchLeft, 2 * kLookVertical);

    // The two queries can return the same object, so collect and dedupe.
    std::vector<GameObject*> found;
    for (auto* arr : {layer->staticObjectsInRect(search, true),
                      layer->damagingObjectsInRect(search, true)}) {
        if (!arr) continue;
        for (auto* obj : CCArrayExt<GameObject*>(arr)) found.push_back(obj);
    }
    std::sort(found.begin(), found.end());
    found.erase(std::unique(found.begin(), found.end()), found.end());

    struct Candidate {
        CCRect rect;
        proto::ObjectKind kind;
    };
    std::vector<Candidate> candidates;
    for (auto* obj : found) {
        auto kind = classify(obj->m_objectType);
        if (!kind) continue;
        CCRect rect = obj->getObjectRect();
        if (rect.getMaxX() < searchLeft) continue;  // already passed
        candidates.push_back({rect, *kind});
    }

    // Nearest first by left edge, capped at kMaxObjects (PROTOCOL.md).
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.rect.getMinX() < b.rect.getMinX();
    });
    size_t n = std::min(candidates.size(), static_cast<size_t>(proto::kMaxObjects));
    for (size_t i = 0; i < n; ++i) {
        const auto& c = candidates[i];
        auto& o = out.objects[i];
        o.x = c.rect.getMidX();
        o.y = c.rect.getMidY() - kGroundY;
        o.w = c.rect.size.width;
        o.h = c.rect.size.height;
        o.kind = static_cast<uint8_t>(c.kind);
    }
    out.numObjects = static_cast<uint16_t>(n);
}

}  // namespace gdrl
