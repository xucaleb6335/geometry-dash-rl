#include "LineServer.hpp"

#include <algorithm>
#include <chrono>
#include <string>
#include <thread>

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
#define GDCTL_CLOSESOCKET closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#define GDCTL_CLOSESOCKET ::close
#endif

namespace gdctl {

namespace {

#ifdef _WIN32
using RawSocket = SOCKET;
#else
using RawSocket = int;
#endif

constexpr intptr_t kInvalid = -1;

RawSocket raw(intptr_t s) { return static_cast<RawSocket>(s); }

bool sendAll(RawSocket s, const std::string& data) {
    size_t sent = 0;
    while (sent < data.size()) {
        int chunk = static_cast<int>(std::min<size_t>(data.size() - sent, 1 << 20));
        auto n = ::send(s, data.data() + sent, chunk, 0);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

}  // namespace

void LineServer::log(const std::string& line) const {
    if (m_logger) m_logger(line);
}

bool LineServer::start(uint16_t port, Handler handler, Logger logger) {
    m_handler = std::move(handler);
    m_logger = std::move(logger);
#ifdef _WIN32
    WSADATA data;
    WSAStartup(MAKEWORD(2, 2), &data);  // never cleaned up: lives until exit
#endif
    RawSocket s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (static_cast<intptr_t>(s) == kInvalid) {
        log("control: socket() failed");
        return false;
    }
#ifdef _WIN32
    // Exclusive: a second game instance must fail to bind, not share the port.
    int yes = 1;
    setsockopt(s, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&yes), sizeof(yes));
#else
    int yes = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
#endif
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // never expose outside the machine
    if (::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || ::listen(s, 8) != 0) {
        log("control: bind/listen failed on port " + std::to_string(port));
        GDCTL_CLOSESOCKET(s);
        return false;
    }
    socklen_t len = sizeof(addr);
    getsockname(s, reinterpret_cast<sockaddr*>(&addr), &len);
    m_port = ntohs(addr.sin_port);
    m_listen = static_cast<intptr_t>(s);
    log("control: listening on 127.0.0.1:" + std::to_string(m_port));
    std::thread([this] { acceptLoop(); }).detach();
    return true;
}

void LineServer::acceptLoop() {
    while (true) {
        RawSocket c = ::accept(raw(m_listen), nullptr, nullptr);
        if (static_cast<intptr_t>(c) == kInvalid) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }
        int yes = 1;
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&yes), sizeof(yes));
        auto sock = static_cast<intptr_t>(c);
        std::thread([this, sock] { clientLoop(sock); }).detach();
    }
}

void LineServer::clientLoop(intptr_t sock) {
    ++m_clients;
    std::string buffer;
    char chunk[65536];
    bool alive = true;
    while (alive) {
        auto n = ::recv(raw(sock), chunk, sizeof(chunk), 0);
        if (n <= 0) break;
        buffer.append(chunk, static_cast<size_t>(n));
        size_t start = 0;
        while (true) {
            auto nl = buffer.find('\n', start);
            if (nl == std::string::npos) break;
            std::string line = buffer.substr(start, nl - start);
            start = nl + 1;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            std::string reply = m_handler(line);
            reply.push_back('\n');
            if (!sendAll(raw(sock), reply)) {
                alive = false;
                break;
            }
        }
        buffer.erase(0, start);
        if (buffer.size() > kMaxLine) {
            sendAll(raw(sock), "{\"ok\":false,\"error\":\"request line too long\",\"code\":\"bad_request\"}\n");
            break;
        }
    }
    GDCTL_CLOSESOCKET(raw(sock));
    --m_clients;
}

}  // namespace gdctl
