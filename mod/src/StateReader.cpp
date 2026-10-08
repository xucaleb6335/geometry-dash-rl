#include "StateReader.hpp"

#include <Geode/Geode.hpp>

using namespace geode::prelude;

namespace gdrl {

void readState(PlayLayer* layer, proto::State& out) {
    out = proto::State{};
    if (!layer) return;

    // LEARN(M1): replace this stub. Start with the player only and check it
    // against the screen with a 1 Hz log, then add objects.
    //
    //   auto* player = layer->m_player1;
    //   out.playerX = player->getPositionX();
    //   out.playerY = ...;
    //
    // Stub values so the pipeline runs end to end before you fill this in.
    out.mode = static_cast<uint8_t>(proto::PlayerMode::Cube);
    out.numObjects = 0;
}

}  // namespace gdrl
