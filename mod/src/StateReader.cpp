#include "StateReader.hpp"

#include <Geode/Geode.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

using namespace geode::prelude;

namespace gdrl {

namespace {

constexpr float kBlock = 30.0f;          // game units per block
constexpr float kLookAhead = 20 * kBlock;  // how far ahead of the player to search
constexpr float kLookVertical = 10 * kBlock;  // how far above and below the player (the encoder ignores more)
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

struct Classified {
    proto::ObjectKind kind;
    proto::ObjectSubtype subtype;
};

// Speed portals are "special" objects in GD; recognise them by object id.
std::optional<proto::ObjectSubtype> speedPortal(int objectID) {
    switch (objectID) {
        case 200: return proto::ObjectSubtype::SpeedSlow;
        case 201: return proto::ObjectSubtype::SpeedNormal;
        case 202: return proto::ObjectSubtype::Speed2x;
        case 203: return proto::ObjectSubtype::Speed3x;
        case 1334: return proto::ObjectSubtype::Speed4x;
        default: return std::nullopt;
    }
}

// Map GD's object type to the protocol's kind and subtype. nullopt = never
// send (decorations, triggers, coins).
std::optional<Classified> classify(GameObject* obj) {
    using K = proto::ObjectKind;
    using S = proto::ObjectSubtype;
    if (auto speed = speedPortal(obj->m_objectID)) return Classified{K::Portal, *speed};
    switch (obj->m_objectType) {
        case GameObjectType::Solid:
        case GameObjectType::Breakable: return Classified{K::Solid, S::Block};
        case GameObjectType::Slope: return Classified{K::Solid, S::Slope};
        case GameObjectType::Hazard: return Classified{K::Hazard, S::Spike};
        case GameObjectType::AnimatedHazard: return Classified{K::Hazard, S::Saw};
        case GameObjectType::YellowJumpRing: return Classified{K::Orb, S::OrbYellow};
        case GameObjectType::PinkJumpRing: return Classified{K::Orb, S::OrbPink};
        case GameObjectType::GravityRing: return Classified{K::Orb, S::OrbGravity};
        case GameObjectType::GreenRing: return Classified{K::Orb, S::OrbGreen};
        case GameObjectType::DropRing: return Classified{K::Orb, S::OrbDrop};
        case GameObjectType::RedJumpRing: return Classified{K::Orb, S::OrbRed};
        case GameObjectType::CustomRing: return Classified{K::Orb, S::OrbCustom};
        case GameObjectType::DashRing: return Classified{K::Orb, S::OrbDash};
        case GameObjectType::GravityDashRing: return Classified{K::Orb, S::OrbGravityDash};
        case GameObjectType::SpiderOrb: return Classified{K::Orb, S::OrbSpider};
        case GameObjectType::TeleportOrb: return Classified{K::Orb, S::OrbTeleport};
        case GameObjectType::YellowJumpPad: return Classified{K::Pad, S::PadYellow};
        case GameObjectType::PinkJumpPad: return Classified{K::Pad, S::PadPink};
        case GameObjectType::GravityPad: return Classified{K::Pad, S::PadGravity};
        case GameObjectType::RedJumpPad: return Classified{K::Pad, S::PadRed};
        case GameObjectType::SpiderPad: return Classified{K::Pad, S::PadSpider};
        case GameObjectType::InverseGravityPortal: return Classified{K::Portal, S::PortalGravityUp};
        case GameObjectType::NormalGravityPortal: return Classified{K::Portal, S::PortalGravityDown};
        case GameObjectType::GravityTogglePortal: return Classified{K::Portal, S::PortalGravityToggle};
        case GameObjectType::ShipPortal: return Classified{K::Portal, S::PortalShip};
        case GameObjectType::CubePortal: return Classified{K::Portal, S::PortalCube};
        case GameObjectType::BallPortal: return Classified{K::Portal, S::PortalBall};
        case GameObjectType::UfoPortal: return Classified{K::Portal, S::PortalUfo};
        case GameObjectType::WavePortal: return Classified{K::Portal, S::PortalWave};
        case GameObjectType::RobotPortal: return Classified{K::Portal, S::PortalRobot};
        case GameObjectType::SpiderPortal: return Classified{K::Portal, S::PortalSpider};
        case GameObjectType::SwingPortal: return Classified{K::Portal, S::PortalSwing};
        case GameObjectType::InverseMirrorPortal: return Classified{K::Portal, S::PortalMirror};
        case GameObjectType::NormalMirrorPortal: return Classified{K::Portal, S::PortalMirrorOff};
        case GameObjectType::RegularSizePortal: return Classified{K::Portal, S::PortalNormalSize};
        case GameObjectType::MiniSizePortal: return Classified{K::Portal, S::PortalMini};
        case GameObjectType::DualPortal: return Classified{K::Portal, S::PortalDual};
        case GameObjectType::SoloPortal: return Classified{K::Portal, S::PortalSolo};
        case GameObjectType::TeleportPortal: return Classified{K::Portal, S::PortalTeleport};
        case GameObjectType::Decoration:
        case GameObjectType::Modifier:
        case GameObjectType::SecretCoin:
        case GameObjectType::UserCoin:
        case GameObjectType::Collectible:
        case GameObjectType::EnterEffectObject:
            return std::nullopt;
        default:
            return Classified{K::Other, S::Unknown};
    }
}

proto::PlayerMode modeOf(PlayerObject* player) {
    // GD has no cube flag: cube is what's left when no other mode flag is set.
    // m_isBird is the UFO and m_isDart is the wave.
    if (player->m_isShip) return proto::PlayerMode::Ship;
    if (player->m_isBall) return proto::PlayerMode::Ball;
    if (player->m_isBird) return proto::PlayerMode::Ufo;
    if (player->m_isDart) return proto::PlayerMode::Wave;
    if (player->m_isRobot) return proto::PlayerMode::Robot;
    if (player->m_isSpider) return proto::PlayerMode::Spider;
    if (player->m_isSwing) return proto::PlayerMode::Swing;
    return proto::PlayerMode::Cube;
}

float velYOf(PlayerObject* player) {
    // m_yVelocity is already world direction (+ = up), upside down included.
    float velY = static_cast<float>(player->m_yVelocity) * kTicksPerVelUnit;
    return player->m_isDart ? velY : velY * kNonWaveVelScale;
}

}  // namespace

void readState(PlayLayer* layer, proto::StateV2& out) {
    out = proto::StateV2{};
    if (!layer) return;

    auto* player = layer->m_player1;
    if (!player) return;

    // Game coordinates: GD's ground is at y = kGroundY. Everything sent is
    // shifted so the ground is at 0 (PROTOCOL.md).
    float gameX = player->getPositionX();
    float gameY = player->getPositionY();
    out.playerX = gameX;
    out.playerY = gameY - kGroundY;

    out.velY = velYOf(player);

    out.xSpeed = xSpeedUnitsPerSecond(player->m_playerSpeed);
    out.levelLength = layer->m_levelLength;

    // cocos rotation is clockwise degrees and never wraps; keep it in [0, 360)
    // like the mock game.
    float rotation = std::fmod(player->getRotation(), 360.0f);
    out.rotation = rotation < 0 ? rotation + 360.0f : rotation;

    if (player->m_isOnGround) out.flags |= proto::kOnGround;
    if (player->m_isUpsideDown) out.flags |= proto::kUpsideDown;
    if (player->m_vehicleSize < 0.99f) out.flags |= proto::kMini;
    out.playerSize = player->getObjectRect().size.width;
    // Corridor bounds (ship/ball/wave/...) are not read yet: TODO after an
    // in-game look at GJGameState::m_portalY and getGroundHeightForMode.
    out.floorY = std::numeric_limits<float>::quiet_NaN();
    out.ceilY = std::numeric_limits<float>::quiet_NaN();
    if (layer->m_gameState.m_isDualMode && layer->m_player2) {
        auto* p2 = layer->m_player2;
        out.flags |= proto::kDual;
        out.p2X = p2->getPositionX();
        out.p2Y = p2->getPositionY() - kGroundY;
        out.p2VelY = velYOf(p2);
        out.p2Mode = static_cast<uint8_t>(modeOf(p2));
        if (p2->m_isOnGround) out.p2Flags |= proto::kOnGround;
        if (p2->m_isUpsideDown) out.p2Flags |= proto::kUpsideDown;
        if (p2->m_vehicleSize < 0.99f) out.p2Flags |= proto::kMini;
    }

    proto::PlayerMode mode = modeOf(player);
    out.mode = static_cast<uint8_t>(mode);

    // Objects: search a box from one block behind the player's left edge to
    // kLookAhead in front, kLookVertical above and below. CCRect is
    // (bottom-left x, bottom-left y, width, height).
    // Setting legacy-view: the object view before 2026-10-13 (no block
    // merging, no No-Touch/speed-portal handling, +-20 blocks vertically),
    // which the Stereo Madness policy was trained on.
    const bool legacy = geode::Mod::get()->getSettingValue<bool>("legacy-view");
    const float lookVertical = legacy ? 20 * kBlock : kLookVertical;
    float playerLeft = player->getObjectRect().getMinX();
    float searchLeft = playerLeft - kBlock;
    CCRect search(searchLeft, gameY - lookVertical,
                  gameX + kLookAhead - searchLeft, 2 * lookVertical);

    // The two queries can return the same object, so collect and dedupe.
    std::vector<GameObject*> found;
    for (auto* arr : {layer->staticObjectsInRect(search, true),
                      layer->damagingObjectsInRect(search, true)}) {
        if (!arr) continue;
        for (auto* obj : CCArrayExt<GameObject*>(arr)) found.push_back(obj);
    }
    // Speed portals are not in the collision sections; scan their own list.
    if (!legacy && layer->m_speedObjects) {
        for (auto* obj : CCArrayExt<GameObject*>(layer->m_speedObjects)) {
            if (obj && search.intersectsRect(obj->getObjectRect())) found.push_back(obj);
        }
    }
    std::sort(found.begin(), found.end());
    found.erase(std::unique(found.begin(), found.end()), found.end());

    struct Candidate {
        CCRect rect;
        Classified cls;
        uint8_t flags;
    };
    std::vector<Candidate> candidates;
    for (auto* obj : found) {
        auto cls = classify(obj);
        if (!cls) continue;
        // Not part of the physics: "No Touch" objects (hazards used as
        // decoration) and objects whose group a toggle trigger turned off.
        // GD's own collision skips them, so they would only mislead the agent.
        if (!legacy && (obj->m_isNoTouch || obj->m_isGroupDisabled)) continue;
        CCRect rect = obj->getObjectRect();
        if (rect.getMaxX() < searchLeft) continue;  // already passed
        uint8_t flags = 0;
        // Opacity is not used: it is visual state that alpha triggers leave
        // behind from the previous attempt (a block faded out last attempt is
        // still at opacity 0 on the next one until its trigger runs again), so
        // the same physics state gave different views (and different block
        // merging) depending on the attempt history. Bloodbath 2026-10-18:
        // after a level reopen the cloned policy died at x 6,095 every time.
        if (legacy ? (obj->m_isHide || obj->getOpacity() == 0) : obj->m_isHide) flags |= proto::kObjHidden;
        // TODO(v2): kObjActivated (used orbs/pads) and kObjMoved, after an in-game look.
        candidates.push_back({rect, *cls, flags});
    }

    // Merge plain blocks into larger rectangles: first stacked columns, then
    // rows of equal columns. Why: only the 64 nearest objects are sent, and on
    // Bloodbath's 3x ship corridors walls of single blocks used them all up
    // within ~150 units (0.3 s of travel), so the agent could not see the
    // corridor ahead. Merged rects keep every outer edge, which is all the
    // collision shape the agent needs.
    if (!legacy) {
        auto same = [](float a, float b) { return std::fabs(a - b) < 0.5f; };
        auto isBlock = [](const Candidate& c) {
            return c.cls.kind == proto::ObjectKind::Solid && c.cls.subtype == proto::ObjectSubtype::Block && c.flags == 0;
        };
        std::vector<Candidate> blocks, rest;
        for (auto& c : candidates) (isBlock(c) ? blocks : rest).push_back(c);
        auto mergePass = [&](bool vertical) {
            std::sort(blocks.begin(), blocks.end(), [&](const Candidate& a, const Candidate& b) {
                if (vertical) {
                    if (!same(a.rect.getMinX(), b.rect.getMinX())) return a.rect.getMinX() < b.rect.getMinX();
                    if (!same(a.rect.getMaxX(), b.rect.getMaxX())) return a.rect.getMaxX() < b.rect.getMaxX();
                    return a.rect.getMinY() < b.rect.getMinY();
                }
                if (!same(a.rect.getMinY(), b.rect.getMinY())) return a.rect.getMinY() < b.rect.getMinY();
                if (!same(a.rect.getMaxY(), b.rect.getMaxY())) return a.rect.getMaxY() < b.rect.getMaxY();
                return a.rect.getMinX() < b.rect.getMinX();
            });
            std::vector<Candidate> merged;
            for (auto& c : blocks) {
                if (!merged.empty()) {
                    auto& m = merged.back();
                    bool joins = vertical
                        ? same(m.rect.getMinX(), c.rect.getMinX()) && same(m.rect.getMaxX(), c.rect.getMaxX())
                              && c.rect.getMinY() <= m.rect.getMaxY() + 0.5f
                        : same(m.rect.getMinY(), c.rect.getMinY()) && same(m.rect.getMaxY(), c.rect.getMaxY())
                              && c.rect.getMinX() <= m.rect.getMaxX() + 0.5f;
                    if (joins) {
                        float x0 = std::min(m.rect.getMinX(), c.rect.getMinX());
                        float y0 = std::min(m.rect.getMinY(), c.rect.getMinY());
                        float x1 = std::max(m.rect.getMaxX(), c.rect.getMaxX());
                        float y1 = std::max(m.rect.getMaxY(), c.rect.getMaxY());
                        m.rect = CCRect(x0, y0, x1 - x0, y1 - y0);
                        continue;
                    }
                }
                merged.push_back(c);
            }
            blocks.swap(merged);
        };
        mergePass(true);
        mergePass(false);
        candidates = std::move(rest);
        candidates.insert(candidates.end(), blocks.begin(), blocks.end());
    }

    // Nearest first by left edge, capped at kMaxObjects (PROTOCOL.md).
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.rect.getMinX() < b.rect.getMinX();
    });
    size_t n = std::min(candidates.size(), static_cast<size_t>(proto::kMaxObjectsV2));
    for (size_t i = 0; i < n; ++i) {
        const auto& c = candidates[i];
        auto& o = out.objects[i];
        o.x = c.rect.getMidX();
        o.y = c.rect.getMidY() - kGroundY;
        o.w = c.rect.size.width;
        o.h = c.rect.size.height;
        o.kind = static_cast<uint8_t>(c.cls.kind);
        o.subtype = static_cast<uint8_t>(c.cls.subtype);
        o.flags = c.flags;
    }
    out.numObjects = static_cast<uint16_t>(n);
}

}  // namespace gdrl
