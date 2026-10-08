#include "Bridge.hpp"

#include <algorithm>
#include <cstring>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using socklen_t = int;
#define GDRL_CLOSESOCKET closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#define GDRL_CLOSESOCKET ::close
#endif

namespace gdrl {

namespace {

constexpr intptr_t kInvalid = -1;

#ifdef _WIN32
using RawSocket = SOCKET;
#else
using RawSocket = int;
#endif

RawSocket raw(intptr_t s) { return static_cast<RawSocket>(s); }

uint32_t expectedLength(proto::MsgType type) {
    switch (type) {
        case proto::MsgType::Hello: return sizeof(proto::Hello);
        case proto::MsgType::Reset: return sizeof(proto::Reset);
        case proto::MsgType::Step: return sizeof(proto::Step);
        case proto::MsgType::Bye: return 0;
        default: return UINT32_MAX;  // not a client-to-server message
    }
}

}  // namespace

Bridge::Bridge() {
#ifdef _WIN32
    WSADATA data;
    WSAStartup(MAKEWORD(2, 2), &data);
#endif
}

Bridge::~Bridge() {
    close();
#ifdef _WIN32
    WSACleanup();
#endif
}

void Bridge::log(const std::string& line) const {
    if (onLog) onLog(line);
}

bool Bridge::listen(uint16_t port) {
    close();
    RawSocket s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (static_cast<intptr_t>(s) == kInvalid) {
        log("bridge: socket() failed");
        return false;
    }
    int yes = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // never expose this outside the machine
    if (::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || ::listen(s, 1) != 0) {
        log("bridge: bind/listen failed on port " + std::to_string(port));
        GDRL_CLOSESOCKET(s);
        return false;
    }
    socklen_t len = sizeof(addr);
    getsockname(s, reinterpret_cast<sockaddr*>(&addr), &len);
    m_port = ntohs(addr.sin_port);
    m_listen = static_cast<intptr_t>(s);
    log("bridge: listening on 127.0.0.1:" + std::to_string(m_port));
    return true;
}

bool Bridge::waitReadable(intptr_t sock, int timeoutMs) const {
    fd_set set;
    FD_ZERO(&set);
    FD_SET(raw(sock), &set);
    timeval tv{};
    timeval* ptv = nullptr;
    if (timeoutMs >= 0) {
        tv.tv_sec = timeoutMs / 1000;
        tv.tv_usec = (timeoutMs % 1000) * 1000;
        ptv = &tv;
    }
    int ready = ::select(static_cast<int>(raw(sock)) + 1, &set, nullptr, nullptr, ptv);
    return ready > 0;
}

bool Bridge::acceptClient(int timeoutMs) {
    if (hasClient()) return true;
    if (m_listen == kInvalid || !waitReadable(m_listen, timeoutMs)) return false;
    RawSocket c = ::accept(raw(m_listen), nullptr, nullptr);
    if (static_cast<intptr_t>(c) == kInvalid) return false;
    int yes = 1;
    setsockopt(c, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&yes), sizeof(yes));
    m_client = static_cast<intptr_t>(c);
    log("bridge: client connected");
    return true;
}

bool Bridge::hasClient() const { return m_client != kInvalid; }

bool Bridge::readExact(void* out, size_t length) {
    auto* p = static_cast<char*>(out);
    size_t got = 0;
    while (got < length) {
        auto n = ::recv(raw(m_client), p + got, static_cast<int>(length - got), 0);
        if (n <= 0) return false;  // 0 = orderly shutdown, <0 = error
        got += static_cast<size_t>(n);
    }
    return true;
}

std::optional<Incoming> Bridge::receive(int timeoutMs) {
    if (!hasClient() || !waitReadable(m_client, timeoutMs)) return std::nullopt;

    proto::Header header{};
    if (!readExact(&header, sizeof(header))) {
        log("bridge: client disconnected");
        dropClient();
        return std::nullopt;
    }
    auto type = static_cast<proto::MsgType>(header.type);
    if (header.magic != proto::kMagic || header.length != expectedLength(type)) {
        sendError(proto::ErrorCode::BadMessage, "bad magic, type or length");
        log("bridge: bad message, dropping client");
        dropClient();
        return std::nullopt;
    }

    switch (type) {
        case proto::MsgType::Hello: {
            proto::Hello msg{};
            if (!readExact(&msg, sizeof(msg))) break;
            return Incoming{msg};
        }
        case proto::MsgType::Reset: {
            proto::Reset msg{};
            if (!readExact(&msg, sizeof(msg))) break;
            return Incoming{msg};
        }
        case proto::MsgType::Step: {
            proto::Step msg{};
            if (!readExact(&msg, sizeof(msg))) break;
            return Incoming{msg};
        }
        case proto::MsgType::Bye:
            return Incoming{ByeMsg{}};
        default:
            break;
    }
    log("bridge: client disconnected mid-message");
    dropClient();
    return std::nullopt;
}

bool Bridge::sendFrame(proto::MsgType type, const void* payload, uint32_t length) {
    if (!hasClient()) return false;
    // One buffer, one send: avoids a tiny header packet followed by the payload.
    std::string buffer(sizeof(proto::Header) + length, '\0');
    proto::Header header{proto::kMagic, proto::kVersion, static_cast<uint16_t>(type), length};
    std::memcpy(buffer.data(), &header, sizeof(header));
    if (length > 0) std::memcpy(buffer.data() + sizeof(header), payload, length);

    size_t sent = 0;
    while (sent < buffer.size()) {
        auto n = ::send(raw(m_client), buffer.data() + sent, static_cast<int>(buffer.size() - sent), 0);
        if (n <= 0) {
            log("bridge: send failed, dropping client");
            dropClient();
            return false;
        }
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool Bridge::sendHelloAck(uint16_t tickRate) {
    proto::HelloAck msg{proto::kVersion, tickRate, proto::kMaxObjects, 0};
    return sendFrame(proto::MsgType::HelloAck, &msg, sizeof(msg));
}

bool Bridge::sendState(const proto::State& state) {
    return sendFrame(proto::MsgType::State, &state, sizeof(state));
}

bool Bridge::sendError(proto::ErrorCode code, std::string_view message) {
    proto::Error msg{};
    msg.code = static_cast<uint16_t>(code);
    std::memcpy(msg.message, message.data(), std::min(message.size(), sizeof(msg.message) - 1));
    return sendFrame(proto::MsgType::Error, &msg, sizeof(msg));
}

void Bridge::dropClient() {
    if (m_client != kInvalid) {
        GDRL_CLOSESOCKET(raw(m_client));
        m_client = kInvalid;
    }
}

void Bridge::close() {
    dropClient();
    if (m_listen != kInvalid) {
        GDRL_CLOSESOCKET(raw(m_listen));
        m_listen = kInvalid;
    }
}

}  // namespace gdrl
