// Localhost TCP server speaking newline-delimited text (one JSON object per
// line). Geode-free on purpose, like mod/src/Bridge.cpp: winsock2.h must be
// included before windows.h, which Geode's forced PCH breaks, so this file is
// built with SKIP_PRECOMPILE_HEADERS.
//
// Threads: one acceptor thread plus one thread per connected client. The
// handler is called on the client's thread and may block (the mod's handler
// queues the command on the game's main thread and waits for its reply).
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace gdctl {

class LineServer {
public:
    // request line (without '\n') -> reply line (without '\n')
    using Handler = std::function<std::string(const std::string&)>;
    using Logger = std::function<void(std::string_view)>;

    // Bind 127.0.0.1:port and start the acceptor thread. Returns false if the
    // port could not be bound. The server lives until the process exits.
    bool start(uint16_t port, Handler handler, Logger logger);

    uint16_t port() const { return m_port; }
    int clients() const { return m_clients.load(); }

    // Longest request line accepted (a level string can be big).
    static constexpr size_t kMaxLine = 64 * 1024 * 1024;

private:
    void acceptLoop();
    void clientLoop(intptr_t sock);
    void log(const std::string& line) const;

    intptr_t m_listen = -1;
    uint16_t m_port = 0;
    Handler m_handler;
    Logger m_logger;
    std::atomic<int> m_clients{0};
};

}  // namespace gdctl
