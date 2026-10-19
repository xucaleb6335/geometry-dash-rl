// Setting show-hitboxes: draws, inside the level, what the agent sees.
//
//   - the hitboxes the bridge reports to the client (readState's object view,
//     the same up to 64 merged rectangles), coloured by kind;
//   - the 8 objects the encoder puts in its slots, outlined and numbered
//     (mirrors gdrl.features.encoder.ObjectListEncoder with the settings every
//     trained policy uses: k = 8, view 600 units ahead and +-300, obstacles first);
//   - the player's hitbox;
//   - the player's path over the last seconds, blue while the agent holds the
//     button and orange while it is released.
//
// For watching and recording only: off by default, and prepare-training turns
// it off. It reads state but never changes it, so it cannot affect physics.
#pragma once

#include <cstddef>
#include <deque>
#include <vector>

class PlayLayer;
namespace cocos2d {
class CCDrawNode;
class CCLabelBMFont;
}

namespace gdrl {

class Overlay {
public:
    static Overlay& get();

    void onLevelStart(PlayLayer* layer);
    void onLevelExit();
    void onReset();                               // a new attempt: forget the path
    void onTick(PlayLayer* layer, bool holding);  // once per physics tick
    void onFrame(PlayLayer* layer);               // once per rendered frame: redraw

private:
    struct PathPoint {
        float x, y;
        bool held;
    };

    void ensureNodes(PlayLayer* layer);
    void hide();

    PlayLayer* m_layer = nullptr;
    cocos2d::CCDrawNode* m_node = nullptr;  // child of the level's object layer
    std::vector<cocos2d::CCLabelBMFont*> m_labels;
    std::deque<PathPoint> m_path;
    bool m_enabled = false;  // the setting, read once per frame
};

}  // namespace gdrl
