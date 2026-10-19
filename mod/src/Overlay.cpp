#include "Overlay.hpp"

#include "StateReader.hpp"

#include <Geode/Geode.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

using namespace geode::prelude;

namespace gdrl {

namespace {

constexpr float kGroundY = 90.0f;  // the protocol's y = 0 is GD's y = 90 (StateReader.cpp)
constexpr float kBlock = 30.0f;
// The encoder settings of every trained policy (configs: object_list, k 8,
// focus obstacles; max_dx 20 blocks, max_dy 10 blocks).
constexpr size_t kSlots = 8;
constexpr float kMaxDx = 20 * kBlock;
constexpr float kMaxDy = 10 * kBlock;
constexpr size_t kPathTicks = 240 * 5;  // the last 5 s of the path

// CCDrawNode blends premultiplied colours.
ccColor4F rgba(float r, float g, float b, float a) { return {r * a, g * a, b * a, a}; }

std::pair<ccColor4F, ccColor4F> kindColors(uint8_t kind) {  // fill, outline
    switch (static_cast<proto::ObjectKind>(kind)) {
        case proto::ObjectKind::Solid: return {rgba(0.35f, 0.65f, 1.0f, 0.10f), rgba(0.35f, 0.65f, 1.0f, 0.95f)};
        case proto::ObjectKind::Hazard: return {rgba(1.0f, 0.25f, 0.30f, 0.20f), rgba(1.0f, 0.25f, 0.30f, 1.0f)};
        case proto::ObjectKind::Orb:
        case proto::ObjectKind::Pad: return {rgba(1.0f, 0.85f, 0.20f, 0.12f), rgba(1.0f, 0.85f, 0.20f, 1.0f)};
        case proto::ObjectKind::Portal: return {rgba(0.30f, 1.0f, 0.55f, 0.10f), rgba(0.30f, 1.0f, 0.55f, 1.0f)};
        default: return {rgba(0.8f, 0.8f, 0.8f, 0.05f), rgba(0.8f, 0.8f, 0.8f, 0.8f)};
    }
}

// The objects in the encoder's slots, in slot order: indices into st.objects.
// Mirrors ObjectListEncoder.encode: in view, sorted by (dx, dy, kind, w, h),
// then obstacles (hazards, and solids reaching above the feet) first.
std::vector<int> slotIndices(const proto::StateV2& st) {
    struct Candidate {
        int i;
        float dx, dy, w, h;
        uint8_t kind;
        bool obstacle;
    };
    const float px = st.playerX, py = st.playerY;
    const bool flipped = (st.flags & proto::kUpsideDown) != 0;
    const float feet = py - kBlock / 2;
    std::vector<Candidate> c;
    for (int i = 0; i < st.numObjects; ++i) {
        const auto& o = st.objects[i];
        const float dx = o.x - px, dy = o.y - py;
        if (o.kind == 0 || !(dx + o.w / 2 > -kBlock / 2) || !(dx - o.w / 2 <= kMaxDx) || !(std::fabs(dy) <= kMaxDy)) continue;
        const float ym = flipped ? 2 * py - o.y : o.y;
        const bool obstacle = o.kind == static_cast<uint8_t>(proto::ObjectKind::Hazard)
            || (o.kind == static_cast<uint8_t>(proto::ObjectKind::Solid) && ym + o.h / 2 > feet + 2.0f);
        c.push_back({i, dx, dy, o.w, o.h, o.kind, obstacle});
    }
    std::stable_sort(c.begin(), c.end(), [](const Candidate& a, const Candidate& b) {
        if (a.dx != b.dx) return a.dx < b.dx;
        if (a.dy != b.dy) return a.dy < b.dy;
        if (a.kind != b.kind) return a.kind < b.kind;
        if (a.w != b.w) return a.w < b.w;
        return a.h < b.h;
    });
    std::stable_partition(c.begin(), c.end(), [](const Candidate& a) { return a.obstacle; });
    std::vector<int> out;
    for (size_t k = 0; k < c.size() && k < kSlots; ++k) out.push_back(c[k].i);
    return out;
}

}  // namespace

Overlay& Overlay::get() {
    static Overlay instance;
    return instance;
}

void Overlay::onLevelStart(PlayLayer* layer) {
    // The nodes of a previous level went away with it.
    m_layer = layer;
    m_node = nullptr;
    m_labels.clear();
    m_path.clear();
}

void Overlay::onLevelExit() {
    m_layer = nullptr;
    m_node = nullptr;
    m_labels.clear();
    m_path.clear();
}

void Overlay::onReset() { m_path.clear(); }

void Overlay::onTick(PlayLayer* layer, bool holding) {
    if (!m_enabled || !layer || layer != m_layer || !layer->m_player1) return;
    auto p = layer->m_player1->getPosition();
    m_path.push_back({p.x, p.y, holding});
    while (m_path.size() > kPathTicks) m_path.pop_front();
}

void Overlay::ensureNodes(PlayLayer* layer) {
    if (m_node || !layer->m_objectLayer) return;
    // In the object layer, so everything is drawn in level coordinates and
    // moves with the camera like the objects themselves.
    m_node = CCDrawNode::create();
    layer->m_objectLayer->addChild(m_node, 9999);
    m_labels.clear();
    for (size_t i = 0; i < kSlots; ++i) {
        auto* label = CCLabelBMFont::create(std::to_string(i + 1).c_str(), "bigFont.fnt");
        label->setScale(0.42f);
        label->setAnchorPoint({0.0f, 0.0f});
        label->setVisible(false);
        layer->m_objectLayer->addChild(label, 10000);
        m_labels.push_back(label);
    }
}

void Overlay::hide() {
    if (m_node) {
        m_node->clear();
        m_node->setVisible(false);
    }
    for (auto* label : m_labels) label->setVisible(false);
}

void Overlay::onFrame(PlayLayer* layer) {
    if (!layer || layer != m_layer) return;
    m_enabled = Mod::get()->getSettingValue<bool>("show-hitboxes");
    if (!m_enabled) {
        hide();
        m_path.clear();
        return;
    }
    ensureNodes(layer);
    auto* player = layer->m_player1;
    if (!m_node || !player) return;
    m_node->setVisible(true);
    m_node->clear();
    for (auto* label : m_labels) label->setVisible(false);

    // What the client gets this frame (the same function builds its STATE).
    proto::StateV2 st{};
    readState(layer, st);
    const ccColor4F none = rgba(0, 0, 0, 0);

    // Every hitbox in the report.
    for (int i = 0; i < st.numObjects; ++i) {
        const auto& o = st.objects[i];
        auto [fill, line] = kindColors(o.kind);
        m_node->drawRect(CCRect(o.x - o.w / 2, o.y + kGroundY - o.h / 2, o.w, o.h), fill, 0.8f, line);
    }

    // The encoder's slots: a white outline and the slot number.
    std::vector<CCPoint> placed;
    auto slots = slotIndices(st);
    for (size_t s = 0; s < slots.size(); ++s) {
        const auto& o = st.objects[slots[s]];
        CCRect r(o.x - o.w / 2, o.y + kGroundY - o.h / 2, o.w, o.h);
        m_node->drawRect(r, none, 1.6f, rgba(1.0f, 1.0f, 1.0f, 0.95f));
        if (s < m_labels.size()) {
            CCPoint at(r.getMinX() - 2.0f, r.getMaxY() + 2.0f);
            for (const auto& q : placed) {  // keep numbers of stacked objects apart
                if (std::fabs(q.x - at.x) < 14.0f && std::fabs(q.y - at.y) < 14.0f) at.y = q.y + 14.0f;
            }
            placed.push_back(at);
            m_labels[s]->setPosition(at);
            m_labels[s]->setVisible(true);
        }
    }

    // The player's path: blue while the button is held, orange while released
    // (the visualizer's colours), fading with age.
    // A dark under-stroke keeps it readable on any level background.
    const size_t n = m_path.size();
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 1; i < n; ++i) {
            const auto& a = m_path[i - 1];
            const auto& b = m_path[i];
            if (std::fabs(b.x - a.x) > 60.0f || std::fabs(b.y - a.y) > 60.0f) continue;  // a respawn
            const float alpha = 0.3f + 0.7f * static_cast<float>(i) / static_cast<float>(n);
            if (pass == 0) {
                m_node->drawSegment(CCPoint(a.x, a.y), CCPoint(b.x, b.y), 2.6f, rgba(0.0f, 0.0f, 0.0f, 0.45f * alpha));
            } else {
                const ccColor4F col = b.held ? rgba(0.30f, 0.72f, 1.0f, alpha) : rgba(0.96f, 0.58f, 0.13f, alpha);
                m_node->drawSegment(CCPoint(a.x, a.y), CCPoint(b.x, b.y), 1.6f, col);
            }
        }
    }

    // The player's own hitbox.
    m_node->drawRect(player->getObjectRect(), none, 1.0f, rgba(0.3f, 1.0f, 0.45f, 1.0f));
}

}  // namespace gdrl
