// TCP transport for the protocol: listen on localhost, accept one client,
// read and write framed messages. Plumbing only; it knows nothing about the
// game. Builds on Windows (Winsock) and POSIX so it can be tested without GD.
#pragma once

#include "Protocol.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace gdrl {

struct ByeMsg {};

// v2: SAVE_STATE, LOAD_STATE or DROP_STATE.
struct SnapshotMsg {
    proto::MsgType type;
    proto::Snapshot msg;
};

// A message the client sent us, already validated against the protocol.
using Incoming = std::variant<proto::Hello, proto::Reset, proto::Step, ByeMsg, SnapshotMsg>;

class Bridge {
public:
    Bridge();
    ~Bridge();
    Bridge(const Bridge&) = delete;
    Bridge& operator=(const Bridge&) = delete;

    // Optional log sink. The mod points this at geode::log; tools use stdout.
    std::function<void(std::string_view)> onLog;

    // Bind 127.0.0.1:port and start listening. Pass 0 to let the OS pick a
    // port, then read it back with port().
    bool listen(uint16_t port);
    uint16_t port() const { return m_port; }

    // Accept a waiting client. timeoutMs = 0 polls without blocking,
    // -1 blocks forever. Returns true if a client is connected afterwards.
    bool acceptClient(int timeoutMs);
    bool hasClient() const;

    // Read one complete message. timeoutMs = 0 polls, -1 blocks.
    // Returns std::nullopt if nothing arrived in time. On a protocol violation
    // it replies ERROR(BAD_MESSAGE), drops the client and returns nullopt.
    // On disconnect it drops the client and returns nullopt.
    std::optional<Incoming> receive(int timeoutMs);

    // The protocol version agreed in HELLO (1 until then, and for each new
    // client). It goes in every header and picks the STATE layout.
    void setVersion(uint16_t version) { m_version = version; }
    uint16_t version() const { return m_version; }

    bool sendHelloAck(uint16_t tickRate);  // v1 ack (standalone server)
    bool sendHelloAck(uint16_t tickRate, uint16_t maxSnapshots);  // in the agreed version
    bool sendState(const proto::State& state);  // v1 layout
    bool sendState(const proto::StateV2& state);  // v2 layout, or its v1 view for a v1 client
    bool sendError(proto::ErrorCode code, std::string_view message);

    void dropClient();
    void close();

private:
    bool sendFrame(proto::MsgType type, const void* payload, uint32_t length);
    bool readExact(void* out, size_t length);
    bool waitReadable(intptr_t sock, int timeoutMs) const;
    void log(const std::string& line) const;

    intptr_t m_listen = -1;
    intptr_t m_client = -1;
    uint16_t m_port = 0;
    uint16_t m_version = 1;
};

}  // namespace gdrl
