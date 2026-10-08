// Turns the live game into a proto::State. This is where you learn how GD
// stores the player and level objects (milestone M1).
#pragma once

#include "Protocol.hpp"

class PlayLayer;  // Geode binding, global namespace

namespace gdrl {

// Fill `out` from the current PlayLayer. Called on the main thread only.
//
// LEARN(M1). What to fill, all in game units (30 units = 1 block):
//   - tick:        your own counter, reset to 0 on RESET (the Controller owns it)
//   - flags:       ON_GROUND, UPSIDE_DOWN, HOLDING (DEAD and COMPLETED come
//                  from the destroyPlayer / levelComplete hooks, not from here)
//   - playerX/Y:   centre of player 1's hitbox
//   - velY, rotation, xSpeed, levelLength, mode
//   - objects:     hitboxes near the player, sorted by left edge, capped at
//                  proto::kMaxObjects. See PROTOCOL.md for which ones to send.
//
// Where to look: layer->m_player1 (PlayerObject), its position and y velocity
// fields; GJBaseGameLayer::staticObjectsInRect / damagingObjectsInRect for
// obstacles; GameObject::getObjectRect for hitboxes; GameObject::m_objectType
// to classify solid vs hazard vs orb. Confirm every field name in the headers
// under $GEODE_SDK/bindings for GD 2.2081 before trusting it.
void readState(PlayLayer* layer, proto::State& out);

}  // namespace gdrl
