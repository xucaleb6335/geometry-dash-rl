#include "Commands.hpp"

#include <Geode/Geode.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace geode::prelude;

namespace gdctl {

namespace {

using Clock = std::chrono::steady_clock;
using Json = matjson::Value;

// ---------------------------------------------------------------------------
// Errors and parameter helpers

struct CmdError {
    std::string code;
    std::string message;
};

[[noreturn]] void fail(std::string code, std::string message) {
    throw CmdError{std::move(code), std::move(message)};
}

Json errorJson(std::string const& code, std::string const& message) {
    Json out = Json::object();
    out["ok"] = false;
    out["code"] = code;
    out["error"] = message;
    return out;
}

Json okJson() {
    Json out = Json::object();
    out["ok"] = true;
    return out;
}

bool has(Json const& req, std::string_view key) {
    return req.contains(key) && !req[key].isNull();
}

std::optional<std::string> optStr(Json const& req, std::string_view key) {
    if (!has(req, key)) return std::nullopt;
    auto r = req[key].asString();
    if (!r) fail("bad_request", std::string(key) + " must be a string");
    return r.unwrap();
}

std::optional<int64_t> optInt(Json const& req, std::string_view key) {
    if (!has(req, key)) return std::nullopt;
    if (auto r = req[key].asInt()) return r.unwrap();
    if (auto d = req[key].asDouble()) return static_cast<int64_t>(d.unwrap());
    if (auto s = req[key].asString()) {
        try {
            return std::stoll(s.unwrap());
        } catch (...) {
        }
    }
    fail("bad_request", std::string(key) + " must be an integer");
}

std::optional<double> optNum(Json const& req, std::string_view key) {
    if (!has(req, key)) return std::nullopt;
    if (auto d = req[key].asDouble()) return d.unwrap();
    if (auto r = req[key].asInt()) return static_cast<double>(r.unwrap());
    fail("bad_request", std::string(key) + " must be a number");
}

std::optional<bool> optBool(Json const& req, std::string_view key) {
    if (!has(req, key)) return std::nullopt;
    if (auto b = req[key].asBool()) return b.unwrap();
    if (auto r = req[key].asInt()) return r.unwrap() != 0;
    if (auto s = req[key].asString()) {
        auto v = s.unwrap();
        if (v == "true" || v == "1" || v == "on" || v == "yes") return true;
        if (v == "false" || v == "0" || v == "off" || v == "no") return false;
    }
    fail("bad_request", std::string(key) + " must be a boolean");
}

std::string reqStr(Json const& req, std::string_view key) {
    auto v = optStr(req, key);
    if (!v) fail("bad_request", "missing parameter: " + std::string(key));
    return *v;
}

std::string toStd(gd::string const& s) { return std::string(s.c_str(), s.size()); }

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// ---------------------------------------------------------------------------
// Frame-based waiting. queueInMainThread runs the function on the next frame.

void waitUntil(std::function<bool()> pred, Clock::time_point deadline, std::function<void(bool)> done) {
    bool ok = false;
    try {
        ok = pred();
    } catch (...) {
        ok = false;
    }
    if (ok) {
        done(true);
        return;
    }
    if (Clock::now() > deadline) {
        done(false);
        return;
    }
    queueInMainThread([pred = std::move(pred), deadline, done = std::move(done)]() mutable {
        waitUntil(std::move(pred), deadline, std::move(done));
    });
}

Clock::time_point after(double seconds) {
    return Clock::now() + std::chrono::milliseconds(static_cast<int64_t>(seconds * 1000));
}

// Run `fn` after `frames` frames.
void afterFrames(int frames, std::function<void()> fn) {
    if (frames <= 0) {
        fn();
        return;
    }
    queueInMainThread([frames, fn = std::move(fn)]() mutable { afterFrames(frames - 1, std::move(fn)); });
}

// ---------------------------------------------------------------------------
// Scene helpers

std::string className(CCObject const* obj) {
    if (!obj) return "null";
    std::string name(geode::cocos::getObjectName(obj));
    for (auto prefix : {"class ", "struct "}) {
        if (name.rfind(prefix, 0) == 0) name = name.substr(std::strlen(prefix));
    }
    return name;
}

CCScene* running() { return CCDirector::get()->getRunningScene(); }

bool isTransition(CCScene* s) { return s && typeinfo_cast<CCTransitionScene*>(s) != nullptr; }

// No transition running or pending.
bool settled() {
    auto s = running();
    return s && !isTransition(s) && CCDirector::get()->getNextScene() == nullptr;
}

std::vector<CCNode*> children(CCNode* node) {
    std::vector<CCNode*> out;
    if (!node || !node->getChildren()) return out;
    for (auto* child : CCArrayExt<CCNode*>(node->getChildren())) out.push_back(child);
    return out;
}

template <class T>
T* findChildOfType(CCNode* node) {
    for (auto* c : children(node)) {
        if (auto* t = typeinfo_cast<T*>(c)) return t;
    }
    return nullptr;
}

PlayLayer* currentPlayLayer() {
    auto s = running();
    if (!s || isTransition(s)) return nullptr;
    return findChildOfType<PlayLayer>(s);
}

LevelEditorLayer* currentEditor() {
    auto s = running();
    if (!s || isTransition(s)) return nullptr;
    return findChildOfType<LevelEditorLayer>(s);
}

// The text a node shows itself (labels only).
std::optional<std::string> ownText(CCNode* node) {
    if (auto* l = typeinfo_cast<CCLabelBMFont*>(node)) return std::string(l->getString());
    if (auto* l = typeinfo_cast<CCLabelTTF*>(node)) return std::string(l->getString());
    return std::nullopt;
}

void collectText(CCNode* node, std::vector<std::string>& out, int depth) {
    if (!node || depth > 8 || !node->isVisible()) return;
    if (auto t = ownText(node)) {
        if (!t->empty()) out.push_back(*t);
        return;  // a label's children are glyph sprites
    }
    for (auto* c : children(node)) collectText(c, out, depth + 1);
}

std::string joinText(CCNode* node) {
    std::vector<std::string> parts;
    collectText(node, parts, 0);
    std::string out;
    for (auto& p : parts) {
        if (!out.empty()) out += " | ";
        out += p;
    }
    return out;
}

// One decimal, computed in double so the JSON has no float noise.
double round1(float v) { return std::round(static_cast<double>(v) * 10.0) / 10.0; }

CCRect worldRect(CCNode* node) {
    auto size = node->getContentSize();
    CCRect local(0, 0, size.width, size.height);
    return CCRectApplyAffineTransform(local, node->nodeToWorldTransform());
}

bool effectiveVisible(CCNode* node) {
    for (auto* n = node; n; n = n->getParent()) {
        if (!n->isVisible()) return false;
    }
    return true;
}

// Path segment for `node` among its parent's children: its ID when unique
// among the siblings, else "Class[i]" (i-th sibling of that class).
std::string segmentFor(CCNode* node) {
    auto* parent = node->getParent();
    std::string id(node->getID());
    auto siblings = children(parent);
    if (!id.empty()) {
        int same = 0;
        for (auto* s : siblings) {
            if (std::string(s->getID()) == id) ++same;
        }
        if (same == 1) return id;
    }
    auto cls = className(node);
    int index = 0;
    for (auto* s : siblings) {
        if (s == node) break;
        if (className(s) == cls) ++index;
    }
    return cls + "[" + std::to_string(index) + "]";
}

std::string pathFor(CCNode* node) {
    std::vector<std::string> segs;
    auto* scene = running();
    for (auto* n = node; n && n != scene && n->getParent(); n = n->getParent()) {
        segs.push_back(segmentFor(n));
    }
    std::string out;
    for (auto it = segs.rbegin(); it != segs.rend(); ++it) {
        if (!out.empty()) out += "/";
        out += *it;
    }
    return out;
}

std::vector<std::string> splitPath(std::string const& path) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : path) {
        if (c == '/') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

CCNode* resolvePath(CCNode* root, std::string const& path) {
    CCNode* node = root;
    for (auto const& seg : splitPath(path)) {
        CCNode* next = nullptr;
        auto kids = children(node);
        auto lb = seg.rfind('[');
        if (!seg.empty() && seg.back() == ']' && lb != std::string::npos) {
            auto cls = seg.substr(0, lb);
            int want = std::atoi(seg.substr(lb + 1, seg.size() - lb - 2).c_str());
            int index = 0;
            for (auto* c : kids) {
                if (className(c) == cls) {
                    if (index == want) {
                        next = c;
                        break;
                    }
                    ++index;
                }
            }
        }
        if (!next) {
            for (auto* c : kids) {
                if (std::string(c->getID()) == seg) {
                    next = c;
                    break;
                }
            }
        }
        if (!next) return nullptr;
        node = next;
    }
    return node;
}

// Find a node from {path} or {id} (recursive ID search). Throws if missing.
CCNode* findTarget(Json const& req) {
    auto* scene = running();
    if (!scene) fail("no_scene", "no running scene");
    if (isTransition(scene)) fail("transitioning", "a scene transition is running; retry shortly");
    if (auto path = optStr(req, "path")) {
        auto* node = resolvePath(scene, *path);
        if (!node) fail("not_found", "no node at path: " + *path);
        return node;
    }
    if (auto id = optStr(req, "node_id")) {
        auto* node = scene->getChildByIDRecursive(*id);
        if (!node) fail("not_found", "no node with id: " + *id);
        return node;
    }
    fail("bad_request", "give path or node_id");
}

void collectOfType(CCNode* node, std::vector<CCNode*>& out, int depth = 0) {
    if (!node || depth > 6) return;
    for (auto* c : children(node)) {
        if (typeinfo_cast<FLAlertLayer*>(c) || typeinfo_cast<GJDropDownLayer*>(c)) {
            out.push_back(c);
            continue;
        }
        // Popups are added to the scene or to a layer near the top; do not
        // descend into game layers (thousands of objects).
        if (typeinfo_cast<GJBaseGameLayer*>(c)) continue;
        collectOfType(c, out, depth + 1);
    }
}

std::vector<CCNode*> findPopups() {
    std::vector<CCNode*> out;
    auto* scene = running();
    if (!scene || isTransition(scene)) return out;
    collectOfType(scene, out);
    // PlayLayer and the editor can host popups as direct children too (above).
    // Topmost first: higher z order wins, then later in the child list.
    std::stable_sort(out.begin(), out.end(), [](CCNode* a, CCNode* b) {
        return a->getZOrder() > b->getZOrder();
    });
    return out;
}

void collectMenuItems(CCNode* node, std::vector<CCMenuItem*>& out, int depth = 0) {
    if (!node || depth > 10) return;
    for (auto* c : children(node)) {
        if (auto* item = typeinfo_cast<CCMenuItem*>(c)) {
            out.push_back(item);
            continue;
        }
        collectMenuItems(c, out, depth + 1);
    }
}

Json popupJson(CCNode* popup) {
    Json p = Json::object();
    p["class"] = className(popup);
    p["path"] = pathFor(popup);
    std::vector<std::string> texts;
    collectText(popup, texts, 0);
    Json arr = Json::array();
    for (auto& t : texts) arr.push(t);
    p["text"] = arr;
    Json buttons = Json::array();
    std::vector<CCMenuItem*> items;
    collectMenuItems(popup, items);
    for (auto* item : items) {
        if (!effectiveVisible(item)) continue;
        Json b = Json::object();
        b["path"] = pathFor(item);
        b["label"] = joinText(item);
        std::string id(item->getID());
        if (!id.empty()) b["id"] = id;
        b["enabled"] = item->isEnabled();
        buttons.push(b);
    }
    p["buttons"] = buttons;
    return p;
}

std::string levelTypeName(GJLevelType t) {
    switch (t) {
        case GJLevelType::Main: return "main";
        case GJLevelType::Editor: return "local";
        case GJLevelType::Saved: return "saved";
        case GJLevelType::SearchResult: return "search";
        default: return "default";
    }
}

Json levelJson(GJGameLevel* level) {
    Json l = Json::object();
    if (!level) return Json(nullptr);
    l["id"] = level->m_levelID.value();
    l["name"] = toStd(level->m_levelName);
    l["type"] = levelTypeName(level->m_levelType);
    l["m_id"] = level->m_M_ID;
    l["has_string"] = !level->m_levelString.empty();
    l["string_length"] = static_cast<int64_t>(level->m_levelString.size());
    l["best_normal"] = level->m_normalPercent.value();
    l["best_practice"] = level->m_practicePercent;
    l["audio_track"] = level->m_audioTrack;
    l["song_id"] = level->m_songID;
    return l;
}

Json sceneJson() {
    Json out = okJson();
    auto* scene = running();
    out["frame"] = static_cast<int64_t>(state().frame);
    if (!scene) {
        out["scene"] = nullptr;
        return out;
    }
    out["scene"] = className(scene);
    out["transitioning"] = isTransition(scene) || CCDirector::get()->getNextScene() != nullptr;
    Json layers = Json::array();
    if (!isTransition(scene)) {
        for (auto* c : children(scene)) layers.push(className(c));
    }
    out["layers"] = layers;
    out["main_layer"] = layers.size() > 0 ? layers[0] : Json(nullptr);

    auto* pl = currentPlayLayer();
    auto* editor = currentEditor();
    out["in_level"] = pl != nullptr;
    out["in_editor"] = editor != nullptr;
    out["level"] = nullptr;
    if (pl) {
        Json l = levelJson(pl->m_level);
        l["attempt"] = pl->m_attempts;
        l["percent"] = static_cast<double>(pl->getCurrentPercent());
        l["practice"] = pl->m_isPracticeMode;
        l["paused"] = pl->m_isPaused || findChildOfType<PauseLayer>(running()) != nullptr;
        l["dead"] = pl->m_player1 ? pl->m_player1->m_isDead : false;
        l["completed"] = pl->m_hasCompletedLevel;
        l["level_length"] = static_cast<double>(pl->m_levelLength);
        if (pl->m_player1) {
            l["player_x"] = static_cast<double>(pl->m_player1->getPositionX());
            l["player_y"] = static_cast<double>(pl->m_player1->getPositionY());
        }
        out["level"] = l;
    } else if (editor) {
        Json l = levelJson(editor->m_level);
        out["level"] = l;
    }
    Json popups = Json::array();
    for (auto* p : findPopups()) popups.push(popupJson(p));
    out["popups"] = popups;
    out["prevent_pause"] = preventPause();
    return out;
}

// ---------------------------------------------------------------------------
// Commands

void cmdPing(Json const&, Reply reply) {
    Json out = okJson();
    out["protocol"] = 1;
    out["mod_version"] = Mod::get()->getVersion().toVString();
    out["geode_version"] = Loader::get()->getVersion().toVString();
    out["gd_version"] = Loader::get()->getGameVersion();
    out["port"] = state().port;
    out["frame"] = static_cast<int64_t>(state().frame);
    out["menu_reached"] = state().menuReached;
    out["prevent_pause"] = preventPause();
    out["pid"] = static_cast<int64_t>(GetCurrentProcessId());
    Json mods = Json::array();
    for (auto* mod : Loader::get()->getAllMods()) {
        Json m = Json::object();
        m["id"] = std::string(mod->getID());
        m["name"] = std::string(mod->getName());
        m["version"] = mod->getVersion().toVString();
        m["enabled"] = mod->isOrWillBeEnabled();
        m["loaded"] = mod->isLoaded();
        m["internal"] = mod->isInternal();
        mods.push(m);
    }
    out["mods"] = mods;
    reply(out);
}

void treeWalk(CCNode* node, std::string const& path, int depth, bool parentVisible, Json const& opts,
              Json& out, int& count, int maxNodes, int maxDepth) {
    if (count >= maxNodes) return;
    bool visible = parentVisible && node->isVisible();
    bool visibleOnly = optBool(opts, "visible_only").value_or(true);
    if (visibleOnly && !visible) return;

    auto* item = typeinfo_cast<CCMenuItem*>(node);
    bool clickable = item && item->isEnabled() && visible;
    std::string cls = className(node);
    std::string id(node->getID());
    auto text = ownText(node);
    std::string label = item ? joinText(item) : "";

    bool include = true;
    if (optBool(opts, "clickable_only").value_or(false) && !clickable) include = false;
    if (auto match = optStr(opts, "match")) {
        auto m = lower(*match);
        auto hay = lower(cls + " " + id + " " + text.value_or("") + " " + label + " " + path);
        if (hay.find(m) == std::string::npos) include = false;
    }
    if (include && depth > 0) {
        Json n = Json::object();
        n["path"] = path;
        n["depth"] = depth;
        n["class"] = cls;
        if (!id.empty()) n["id"] = id;
        if (node->getTag() != -1) n["tag"] = node->getTag();
        n["visible"] = visible;
        auto r = worldRect(node);
        n["x"] = round1(r.origin.x);
        n["y"] = round1(r.origin.y);
        n["w"] = round1(r.size.width);
        n["h"] = round1(r.size.height);
        if (text) n["text"] = *text;
        if (item) {
            n["label"] = label;
            n["clickable"] = clickable;
        }
        if (auto* toggler = typeinfo_cast<CCMenuItemToggler*>(node)) n["toggled"] = toggler->isToggled();
        n["children"] = static_cast<int64_t>(node->getChildrenCount());
        out.push(n);
        ++count;
    }
    if (depth >= maxDepth) return;
    // Labels' children are glyphs; game layers have thousands of objects.
    if (text) return;
    if (typeinfo_cast<CCSpriteBatchNode*>(node) && depth > 0) return;
    for (auto* c : children(node)) {
        auto seg = segmentFor(c);
        treeWalk(c, path.empty() ? seg : path + "/" + seg, depth + 1, visible, opts, out, count, maxNodes, maxDepth);
    }
}

void cmdTree(Json const& req, Reply reply) {
    auto* scene = running();
    if (!scene) fail("no_scene", "no running scene");
    if (isTransition(scene)) fail("transitioning", "a scene transition is running; retry shortly");
    CCNode* root = scene;
    std::string rootPath;
    if (auto path = optStr(req, "root")) {
        root = resolvePath(scene, *path);
        if (!root) fail("not_found", "no node at path: " + *path);
        rootPath = *path;
        while (!rootPath.empty() && rootPath.back() == '/') rootPath.pop_back();
    }
    int maxDepth = static_cast<int>(optInt(req, "depth").value_or(12));
    int maxNodes = static_cast<int>(optInt(req, "max_nodes").value_or(3000));
    Json nodes = Json::array();
    int count = 0;
    // The root itself is depth 0 and not listed; children start at depth 1.
    bool rootVisible = effectiveVisible(root);
    treeWalk(root, rootPath, 0, rootVisible || root == scene, req, nodes, count, maxNodes, maxDepth);
    Json out = okJson();
    out["scene"] = className(scene);
    auto win = CCDirector::get()->getWinSize();
    out["win_w"] = static_cast<double>(win.width);
    out["win_h"] = static_cast<double>(win.height);
    out["count"] = count;
    out["truncated"] = count >= maxNodes;
    out["nodes"] = nodes;
    reply(out);
}

// Window frame pixels (what CCEGLView's touch handlers take) for a point in
// cocos coordinates (points, origin bottom-left).
CCPoint toFrame(CCPoint gl) {
    auto* view = CCEGLView::sharedOpenGLView();
    auto vp = view->getViewPortRect();
    auto win = CCDirector::get()->getWinSize();
    return CCPoint(gl.x * view->getScaleX() + vp.origin.x, (win.height - gl.y) * view->getScaleY() + vp.origin.y);
}

void touchAt(CCPoint gl, std::function<void()> done) {
    auto frame = toFrame(gl);
    auto* view = CCEGLView::sharedOpenGLView();
    int id = 0;
    float xs[1] = {frame.x};
    float ys[1] = {frame.y};
    view->handleTouchesBegin(1, &id, xs, ys, 0.0);
    afterFrames(1, [frame, done = std::move(done)] {
        auto* view = CCEGLView::sharedOpenGLView();
        int id = 0;
        float xs[1] = {frame.x};
        float ys[1] = {frame.y};
        view->handleTouchesEnd(1, &id, xs, ys, 0.0);
        done();
    });
}

Json nodeBrief(CCNode* node) {
    Json n = Json::object();
    n["path"] = pathFor(node);
    n["class"] = className(node);
    std::string id(node->getID());
    if (!id.empty()) n["id"] = id;
    if (auto* item = typeinfo_cast<CCMenuItem*>(node)) n["label"] = joinText(item);
    return n;
}

bool isDescendant(CCNode* node, CCNode* ancestor) {
    for (auto* n = node; n; n = n->getParent()) {
        if (n == ancestor) return true;
    }
    return false;
}

void cmdClick(Json const& req, Reply reply) {
    auto win = CCDirector::get()->getWinSize();
    std::optional<CCPoint> point;
    if (has(req, "x") && has(req, "y")) {
        point = CCPoint(static_cast<float>(*optNum(req, "x")), static_cast<float>(*optNum(req, "y")));
    } else if (has(req, "fx") && has(req, "fy")) {
        // Fractions of the window, origin top-left (like a screenshot).
        point = CCPoint(static_cast<float>(*optNum(req, "fx") * win.width),
                        static_cast<float>((1.0 - *optNum(req, "fy")) * win.height));
    }
    if (point) {
        if (!settled()) fail("transitioning", "a scene transition is running; retry shortly");
        auto p = *point;
        touchAt(p, [reply, p] {
            Json out = okJson();
            out["method"] = "touch";
            out["x"] = static_cast<double>(p.x);
            out["y"] = static_cast<double>(p.y);
            reply(out);
        });
        return;
    }

    auto* node = findTarget(req);
    bool force = optBool(req, "force").value_or(false);
    if (!force && !effectiveVisible(node)) fail("not_visible", "node is not visible (pass force to click anyway)");

    Json out = okJson();
    out["node"] = nodeBrief(node);
    auto popups = findPopups();
    if (!popups.empty() && !isDescendant(node, popups.front())) {
        out["warning"] = "a popup is open and this node is not in it; real input would hit the popup";
    }
    bool useTouch = optBool(req, "touch").value_or(false);
    if (auto* item = typeinfo_cast<CCMenuItem*>(node); item && !useTouch) {
        if (!item->isEnabled() && !force) fail("disabled", "menu item is disabled");
        out["method"] = "activate";
        // activate() runs the callback, like a real click (sound included).
        item->retain();
        item->activate();
        item->release();
        reply(out);
        return;
    }
    auto r = worldRect(node);
    CCPoint center(r.getMidX(), r.getMidY());
    out["method"] = "touch";
    out["x"] = static_cast<double>(center.x);
    out["y"] = static_cast<double>(center.y);
    touchAt(center, [reply, out] { reply(out); });
}

std::optional<enumKeyCodes> keyCode(std::string name) {
    name = lower(name);
    static const std::map<std::string, int> named = {
        {"space", 0x20}, {"enter", 0x0D}, {"return", 0x0D}, {"escape", 0x1B}, {"esc", 0x1B},
        {"tab", 0x09}, {"backspace", 0x08}, {"shift", 0x10}, {"ctrl", 0x11}, {"control", 0x11},
        {"alt", 0x12}, {"left", 0x25}, {"up", 0x26}, {"right", 0x27}, {"down", 0x28},
        {"delete", 0x2E}, {"home", 0x24}, {"end", 0x23}, {"pageup", 0x21}, {"pagedown", 0x22},
    };
    if (auto it = named.find(name); it != named.end()) return static_cast<enumKeyCodes>(it->second);
    if (name.size() == 1 && std::isalnum(static_cast<unsigned char>(name[0]))) {
        return static_cast<enumKeyCodes>(std::toupper(static_cast<unsigned char>(name[0])));
    }
    if (name.size() >= 2 && name[0] == 'f') {
        int n = std::atoi(name.c_str() + 1);
        if (n >= 1 && n <= 12) return static_cast<enumKeyCodes>(0x70 + n - 1);
    }
    return std::nullopt;
}

void cmdKey(Json const& req, Reply reply) {
    auto name = reqStr(req, "key");
    auto action = optStr(req, "action").value_or("tap");
    if (action != "tap" && action != "down" && action != "up") fail("bad_request", "action must be tap, down or up");
    Json out = okJson();
    out["key"] = name;
    out["action"] = action;
    if (lower(name) == "back") {
        // GD's ESC/back handling goes through the keypad dispatcher.
        bool handled = CCDirector::get()->getKeypadDispatcher()->dispatchKeypadMSG(kTypeBackClicked);
        out["handled"] = handled;
        reply(out);
        return;
    }
    std::optional<enumKeyCodes> code;
    if (auto c = optInt(req, "code")) code = static_cast<enumKeyCodes>(*c);
    else code = keyCode(name);
    if (!code) fail("bad_request", "unknown key: " + name + " (use a letter, digit, space, enter, escape, back, arrows, f1-f12 or code)");
    auto* kd = CCDirector::get()->getKeyboardDispatcher();
    auto c = *code;
    if (action == "down" || action == "tap") kd->dispatchKeyboardMSG(c, true, false, 0.0);
    if (action == "up") kd->dispatchKeyboardMSG(c, false, false, 0.0);
    if (action == "tap") {
        int holdFrames = static_cast<int>(optInt(req, "frames").value_or(2));
        afterFrames(holdFrames, [c, reply, out] {
            CCDirector::get()->getKeyboardDispatcher()->dispatchKeyboardMSG(c, false, false, 0.0);
            reply(out);
        });
        return;
    }
    reply(out);
}

void cmdScreenshot(Json const& req, Reply reply) {
    auto path = reqStr(req, "path");
    int targetWidth = static_cast<int>(optInt(req, "width").value_or(960));
    auto* scene = running();
    if (!scene) fail("no_scene", "no running scene");

    auto win = CCDirector::get()->getWinSize();
    auto* rt = CCRenderTexture::create(static_cast<int>(win.width), static_cast<int>(win.height),
                                       kCCTexture2DPixelFormat_RGBA8888);
    if (!rt) fail("render_failed", "could not create a render texture");
    rt->begin();
    scene->visit();
    rt->end();
    CCImage* img = rt->newCCImage(true);
    if (!img) fail("render_failed", "could not read the render texture");
    int w = img->getWidth();
    int h = img->getHeight();
    auto* src = img->getData();
    int outW = w, outH = h;
    if (targetWidth > 0 && targetWidth < w) {
        outW = targetWidth;
        outH = std::max(1, static_cast<int>(std::lround(static_cast<double>(h) * outW / w)));
    }
    // Box-filter downscale to RGBA, drop alpha later (saveToFile toRGB).
    std::vector<unsigned char> pixels(static_cast<size_t>(outW) * outH * 4);
    for (int y = 0; y < outH; ++y) {
        int y0 = y * h / outH, y1 = std::max(y0 + 1, (y + 1) * h / outH);
        for (int x = 0; x < outW; ++x) {
            int x0 = x * w / outW, x1 = std::max(x0 + 1, (x + 1) * w / outW);
            unsigned sum[4] = {0, 0, 0, 0};
            int n = 0;
            for (int yy = y0; yy < y1; ++yy) {
                for (int xx = x0; xx < x1; ++xx) {
                    auto* p = src + (static_cast<size_t>(yy) * w + xx) * 4;
                    for (int k = 0; k < 4; ++k) sum[k] += p[k];
                    ++n;
                }
            }
            auto* q = pixels.data() + (static_cast<size_t>(y) * outW + x) * 4;
            for (int k = 0; k < 3; ++k) q[k] = static_cast<unsigned char>(sum[k] / n);
            q[3] = 255;
        }
    }
    img->release();

    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    auto* outImg = new CCImage();
    bool ok = outImg->initWithImageData(pixels.data(), static_cast<int>(pixels.size()), CCImage::kFmtRawData,
                                        outW, outH, 8) &&
              outImg->saveToFile(path.c_str(), true);
    outImg->release();
    if (!ok) fail("save_failed", "could not write PNG to " + path);
    Json out = okJson();
    out["path"] = path;
    out["width"] = outW;
    out["height"] = outH;
    out["win_w"] = static_cast<double>(win.width);
    out["win_h"] = static_cast<double>(win.height);
    out["points_per_pixel"] = static_cast<double>(win.width) / outW;
    out["scene"] = className(scene);
    reply(out);
}

// ----- levels

std::vector<GJGameLevel*> localLevels() {
    std::vector<GJGameLevel*> out;
    auto* arr = LocalLevelManager::sharedState()->m_localLevels;
    if (!arr) return out;
    for (auto* l : CCArrayExt<GJGameLevel*>(arr)) out.push_back(l);
    return out;
}

constexpr int kMainLevelIds[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                 5001, 5002, 5003, 5004, 3001};

GJGameLevel* resolveLevel(Json const& req) {
    auto type = lower(optStr(req, "type").value_or("main"));
    auto* glm = GameLevelManager::sharedState();
    if (type == "main") {
        auto id = optInt(req, "id");
        if (!id) fail("bad_request", "main levels need id (1 = Stereo Madness)");
        auto* level = glm->getMainLevel(static_cast<int>(*id), false);
        if (!level || level->m_levelString.empty()) fail("not_found", "no main level " + std::to_string(*id));
        return level;
    }
    if (type == "local" || type == "editor" || type == "created") {
        auto levels = localLevels();
        if (auto idx = optInt(req, "index")) {
            if (*idx < 0 || *idx >= static_cast<int64_t>(levels.size())) fail("not_found", "local index out of range");
            return levels[*idx];
        }
        if (auto id = optInt(req, "id")) {
            for (auto* l : levels) {
                if (l->m_M_ID == *id || (l->m_levelID.value() != 0 && l->m_levelID.value() == *id)) return l;
            }
            fail("not_found", "no local level with m_id/id " + std::to_string(*id));
        }
        if (auto name = optStr(req, "name")) {
            std::vector<GJGameLevel*> matches;
            for (auto* l : levels) {
                if (toStd(l->m_levelName) == *name) matches.push_back(l);
            }
            if (matches.empty()) fail("not_found", "no local level named " + *name);
            if (matches.size() > 1) fail("ambiguous", "several local levels named " + *name + "; use id (m_id) or index");
            return matches.front();
        }
        fail("bad_request", "local levels need name, id (m_id) or index");
    }
    if (type == "saved" || type == "online") {
        auto id = optInt(req, "id");
        if (!id) fail("bad_request", "saved levels need id");
        auto* level = glm->getSavedLevel(static_cast<int>(*id));
        if (!level) fail("not_found", "level " + std::to_string(*id) + " is not saved locally (download it in game first)");
        if (level->m_levelString.empty()) fail("not_downloaded", "level " + std::to_string(*id) + " has no level data; download it first");
        return level;
    }
    fail("bad_request", "type must be main, local or saved");
}

void goToMenu(double timeout, std::function<void(bool, std::string)> done);

// Quit the current level (if any) and wait until the game settles.
void leaveLevel(double timeout, std::function<void(bool, std::string)> done) {
    auto* pl = currentPlayLayer();
    if (!pl) {
        waitUntil(settled, after(timeout), [done](bool ok) { done(ok, ok ? "" : "scene did not settle"); });
        return;
    }
    pl->onQuit();
    waitUntil([] { return settled() && !currentPlayLayer() && PlayLayer::get() == nullptr; }, after(timeout),
              [done](bool ok) { done(ok, ok ? "" : "level did not close in time"); });
}

void cmdOpenLevel(Json const& req, Reply reply) {
    auto* level = resolveLevel(req);
    bool practice = optBool(req, "practice").value_or(false);
    double timeout = optNum(req, "wait").value_or(20.0);
    if (currentEditor()) fail("in_editor", "the editor is open; run menu with save=true first");
    level->retain();
    leaveLevel(timeout, [level, practice, timeout, reply](bool ok, std::string err) {
        if (!ok) {
            level->release();
            reply(errorJson("timeout", err));
            return;
        }
        // Like LevelPage/LevelInfoLayer: stop the menu music, then start.
        FMODAudioEngine::get()->stopAllMusic(true);
        auto* scene = PlayLayer::scene(level, false, false);
        level->release();
        if (!scene) {
            reply(errorJson("open_failed", "PlayLayer::scene returned null"));
            return;
        }
        CCDirector::get()->replaceScene(CCTransitionFade::create(0.5f, scene));
        waitUntil([scene] { return settled() && running() == scene && currentPlayLayer() != nullptr; },
                  after(timeout), [practice, reply](bool ok) {
                      if (!ok) {
                          reply(errorJson("timeout", "level did not finish opening"));
                          return;
                      }
                      auto* pl = currentPlayLayer();
                      if (pl->m_isPracticeMode != practice) pl->togglePracticeMode(practice);
                      // One more frame so the level has updated once.
                      afterFrames(2, [reply] { reply(sceneJson()); });
                  });
    });
}

void cmdExitLevel(Json const& req, Reply reply) {
    bool toMenu = optBool(req, "menu").value_or(false);
    double timeout = optNum(req, "wait").value_or(20.0);
    if (!currentPlayLayer()) fail("not_in_level", "no level is open");
    leaveLevel(timeout, [toMenu, timeout, reply](bool ok, std::string err) {
        if (!ok) {
            reply(errorJson("timeout", err));
            return;
        }
        if (!toMenu) {
            reply(sceneJson());
            return;
        }
        goToMenu(timeout, [reply](bool ok, std::string err) {
            if (!ok) reply(errorJson("timeout", err));
            else reply(sceneJson());
        });
    });
}

bool atMenu() {
    auto* s = running();
    return settled() && s && findChildOfType<MenuLayer>(s) != nullptr;
}

void goToMenu(double timeout, std::function<void(bool, std::string)> done) {
    if (atMenu()) {
        done(true, "");
        return;
    }
    if (currentPlayLayer()) {
        leaveLevel(timeout, [timeout, done](bool ok, std::string err) {
            if (!ok) done(false, err);
            else goToMenu(timeout, done);
        });
        return;
    }
    if (!settled()) {
        waitUntil(settled, after(timeout), [timeout, done](bool ok) {
            if (!ok) done(false, "scene did not settle");
            else goToMenu(timeout, done);
        });
        return;
    }
    auto* scene = MenuLayer::scene(false);
    CCDirector::get()->replaceScene(CCTransitionFade::create(0.5f, scene));
    waitUntil([scene] { return settled() && running() == scene; }, after(timeout),
              [done](bool ok) { done(ok, ok ? "" : "menu did not open in time"); });
}

void cmdMenu(Json const& req, Reply reply) {
    double timeout = optNum(req, "wait").value_or(20.0);
    if (auto* editor = currentEditor()) {
        if (!optBool(req, "save").value_or(false)) {
            fail("in_editor", "the editor is open; pass save=true to save and exit it (there is no discard)");
        }
        auto* pause = EditorPauseLayer::create(editor);
        pause->onSaveAndExit(nullptr);
        waitUntil([] { return settled() && !currentEditor(); }, after(timeout), [timeout, reply](bool ok) {
            if (!ok) {
                reply(errorJson("timeout", "editor did not close"));
                return;
            }
            goToMenu(timeout, [reply](bool ok, std::string err) {
                if (!ok) reply(errorJson("timeout", err));
                else reply(sceneJson());
            });
        });
        return;
    }
    goToMenu(timeout, [reply](bool ok, std::string err) {
        if (!ok) reply(errorJson("timeout", err));
        else reply(sceneJson());
    });
}

void cmdClosePopups(Json const& req, Reply reply) {
    auto popups = findPopups();
    Json closed = Json::array();
    // Kept retained until they leave the scene (drop-downs animate out).
    auto pending = std::make_shared<std::vector<CCNode*>>();
    for (auto* p : popups) {
        closed.push(className(p));
        p->retain();
        pending->push_back(p);
        if (auto* alert = typeinfo_cast<FLAlertLayer*>(p)) alert->keyBackClicked();
        else if (auto* drop = typeinfo_cast<GJDropDownLayer*>(p)) drop->keyBackClicked();
    }
    double wait = optNum(req, "wait").value_or(3.0);
    waitUntil(
        [pending] {
            return std::all_of(pending->begin(), pending->end(), [](CCNode* n) { return n->getParent() == nullptr; });
        },
        after(wait), [reply, closed, pending](bool) {
            for (auto* p : *pending) p->release();
            Json out = okJson();
            out["closed"] = closed;
            Json remaining = Json::array();
            for (auto* p : findPopups()) remaining.push(popupJson(p));
            out["remaining"] = remaining;
            reply(out);
        });
}

void cmdListLevels(Json const& req, Reply reply) {
    auto types = lower(optStr(req, "types").value_or("main,local,saved"));
    Json out = okJson();
    auto* glm = GameLevelManager::sharedState();
    if (types.find("main") != std::string::npos) {
        Json arr = Json::array();
        for (int id : kMainLevelIds) {
            auto* l = glm->getMainLevel(id, true);
            if (!l) continue;
            auto name = toStd(l->m_levelName);
            if (name.empty()) continue;
            Json j = Json::object();
            j["id"] = id;
            j["name"] = name;
            j["best_normal"] = l->m_normalPercent.value();
            arr.push(j);
        }
        out["main"] = arr;
    }
    if (types.find("local") != std::string::npos) {
        Json arr = Json::array();
        int index = 0;
        for (auto* l : localLevels()) {
            Json j = levelJson(l);
            j["index"] = index++;
            arr.push(j);
        }
        out["local"] = arr;
    }
    if (types.find("saved") != std::string::npos) {
        Json arr = Json::array();
        if (auto* saved = glm->getSavedLevels(false, 0)) {
            for (auto* l : CCArrayExt<GJGameLevel*>(saved)) arr.push(levelJson(l));
        }
        out["saved"] = arr;
    }
    reply(out);
}

void cmdCreateLevel(Json const& req, Reply reply) {
    auto name = reqStr(req, "name");
    auto levelString = reqStr(req, "level_string");
    bool replace = optBool(req, "replace").value_or(false);
    bool save = optBool(req, "save").value_or(true);
    if (name.empty() || name.size() > 20) fail("bad_request", "name must be 1 to 20 characters (GD's limit)");

    auto* glm = GameLevelManager::sharedState();
    for (auto* l : localLevels()) {
        if (toStd(l->m_levelName) != name) continue;
        if (!replace) fail("exists", "a local level named " + name + " exists (pass replace=true)");
        glm->deleteLevel(l);
    }
    // GD stores level data gzip+base64 ("H4sI..."). Accept either form.
    gd::string data(levelString.c_str());
    if (levelString.rfind("H4sI", 0) != 0) data = ZipUtils::compressString(data, false, 0);

    auto* level = glm->createNewLevel();
    if (!level) fail("create_failed", "createNewLevel returned null");
    level->m_levelName = gd::string(name.c_str());
    level->m_levelString = data;
    if (auto track = optInt(req, "audio_track")) level->m_audioTrack = static_cast<int>(*track);
    if (auto song = optInt(req, "song_id")) level->m_songID = static_cast<int>(*song);
    if (save) LocalLevelManager::sharedState()->save();

    Json out = okJson();
    out["level"] = levelJson(level);
    int index = 0;
    for (auto* l : localLevels()) {
        if (l == level) break;
        ++index;
    }
    out["index"] = index;
    out["saved"] = save;
    reply(out);
}

void cmdDeleteLevel(Json const& req, Reply reply) {
    Json r = req;
    r["type"] = "local";
    auto* level = resolveLevel(r);
    auto name = toStd(level->m_levelName);
    auto prefix = optStr(req, "prefix").value_or("gdrl-test-");
    if (name.rfind(prefix, 0) != 0 && !optBool(req, "force").value_or(false)) {
        fail("refused", "only levels named " + prefix + "* are deleted without force=true");
    }
    if (currentPlayLayer() && currentPlayLayer()->m_level == level) fail("in_use", "exit the level first");
    if (currentEditor()) fail("in_editor", "close the editor first");
    Json out = okJson();
    out["deleted"] = levelJson(level);
    GameLevelManager::sharedState()->deleteLevel(level);
    if (optBool(req, "save").value_or(true)) LocalLevelManager::sharedState()->save();
    reply(out);
}

void cmdSetModEnabled(Json const& req, Reply reply) {
    auto id = reqStr(req, "mod");
    auto enabled = optBool(req, "enabled");
    if (!enabled) fail("bad_request", "missing parameter: enabled");
    auto* mod = Loader::get()->getInstalledMod(id);
    if (!mod) fail("not_found", "mod not installed: " + id);
    if (mod->isInternal()) fail("refused", "cannot toggle the loader");
    if (mod == Mod::get() && !*enabled) fail("refused", "cannot disable the control mod through itself");
    auto res = *enabled ? mod->enable() : mod->disable();
    if (!res) fail("geode_error", res.unwrapErr());
    Json out = okJson();
    out["mod"] = id;
    out["enabled"] = mod->isOrWillBeEnabled();
    out["loaded"] = mod->isLoaded();
    out["restart_required"] = mod->isOrWillBeEnabled() != mod->isLoaded();
    reply(out);
}

// Set a bool setting of a loaded mod at runtime (e.g. the bridge's
// watch-mode and legacy-view, which it reads every frame).
void cmdSetModSetting(Json const& req, Reply reply) {
    auto id = reqStr(req, "mod");
    auto key = reqStr(req, "key");
    auto value = optBool(req, "value");
    if (!value) fail("bad_request", "missing parameter: value (bool)");
    auto* mod = Loader::get()->getLoadedMod(id);
    if (!mod) fail("not_found", "mod not loaded: " + id);
    if (!mod->hasSetting(key)) fail("not_found", "no setting " + key + " in " + id);
    mod->setSettingValue<bool>(key, *value);
    Json out = okJson();
    out["mod"] = id;
    out["key"] = key;
    out["value"] = mod->getSettingValue<bool>(key);
    reply(out);
}

void cmdQuit(Json const& req, Reply reply) {
    bool save = optBool(req, "save").value_or(true);
    bool restart = optBool(req, "restart").value_or(false);
    Json out = okJson();
    out["saving"] = save;
    out["restart"] = restart;
    reply(out);
    // Give the reply a moment to flush before the process goes away.
    afterFrames(5, [save, restart] {
        if (restart) geode::utils::game::restart(save);
        else geode::utils::game::exit(save);
    });
}

void cmdPreventPause(Json const& req, Reply reply) {
    if (auto enabled = optBool(req, "enabled")) {
        state().preventPauseOverride = *enabled;
        if (optBool(req, "persist").value_or(false)) Mod::get()->setSettingValue<bool>("prevent-pause", *enabled);
    }
    Json out = okJson();
    out["prevent_pause"] = preventPause();
    out["setting"] = Mod::get()->getSettingValue<bool>("prevent-pause");
    reply(out);
}

// Named GameManager variables (GD's own keys; any "0xxx" key also works).
const std::map<std::string, std::string>& optionAliases() {
    static const std::map<std::string, std::string> aliases = {
        {"show_percent", "0040"},
        {"auto_retry", "0026"},
        {"auto_checkpoints", "0027"},
        {"show_cursor", "0024"},
        {"fast_menu", "0168"},
    };
    return aliases;
}

Json optionsJson() {
    Json out = Json::object();
    auto* gm = GameManager::sharedState();
    for (auto const& [name, key] : optionAliases()) out[name] = gm->getGameVariable(key.c_str());
    out["fps_target"] = static_cast<double>(gm->m_customFPSTarget);
    out["animation_interval"] = CCDirector::get()->getAnimationInterval();
    out["fps_now"] = 1.0 / std::max(1e-6, CCDirector::get()->getAnimationInterval());
    out["vsync"] = gm->m_vsyncEnabled;
    return out;
}

void cmdGetOption(Json const& req, Reply reply) {
    Json out = okJson();
    if (auto key = optStr(req, "key")) {
        auto k = *key;
        if (auto it = optionAliases().find(k); it != optionAliases().end()) k = it->second;
        out["key"] = k;
        out["value"] = GameManager::sharedState()->getGameVariable(k.c_str());
    }
    out["options"] = optionsJson();
    Json aliases = Json::object();
    for (auto const& [name, key] : optionAliases()) aliases[name] = key;
    out["aliases"] = aliases;
    reply(out);
}

void cmdSetOption(Json const& req, Reply reply) {
    auto key = reqStr(req, "key");
    Json out = okJson();
    if (key == "fps") {
        auto fps = optNum(req, "value");
        if (!fps || *fps < 1 || *fps > 2000) fail("bad_request", "fps value must be 1 to 2000");
        CCDirector::get()->setAnimationInterval(1.0 / *fps);
        out["note"] = "applies now; GD may reset it when video settings change";
    } else {
        auto value = optBool(req, "value");
        if (!value) fail("bad_request", "value must be a boolean");
        auto k = key;
        if (auto it = optionAliases().find(k); it != optionAliases().end()) k = it->second;
        if (k.size() != 4 || !std::all_of(k.begin(), k.end(), ::isdigit)) {
            fail("bad_request", "key must be an alias, fps, or a 4-digit GameManager variable like 0040");
        }
        GameManager::sharedState()->setGameVariable(k.c_str(), *value);
        out["key"] = k;
        out["value"] = *value;
    }
    out["options"] = optionsJson();
    reply(out);
}

// ----- online levels, songs, level strings

std::string difficultyName(GJGameLevel* l) {
    if (l->m_demon.value() != 0) {
        switch (l->m_demonDifficulty) {
            case 3: return "easy demon";
            case 4: return "medium demon";
            case 5: return "insane demon";
            case 6: return "extreme demon";
            default: return "hard demon";
        }
    }
    if (l->m_autoLevel) return "auto";
    switch (static_cast<int>(l->m_difficulty)) {
        case 1: return "easy";
        case 2: return "normal";
        case 3: return "hard";
        case 4: return "harder";
        case 5: return "insane";
        default: return "na";
    }
}

Json onlineLevelJson(GJGameLevel* l) {
    Json j = levelJson(l);
    j["creator"] = toStd(l->m_creatorName);
    j["difficulty"] = difficultyName(l);
    j["stars"] = l->m_stars.value();
    j["demon"] = l->m_demon.value() != 0;
    j["downloads"] = l->m_downloads;
    j["likes"] = l->m_likes;
    j["length"] = l->m_levelLength;  // 0 tiny .. 4 XL, 5 platformer
    j["objects"] = l->m_objectCount.value();
    j["original"] = l->m_originalLevel.value();
    return j;
}

void cmdSearchLevels(Json const& req, Reply reply) {
    auto query = reqStr(req, "query");
    int page = static_cast<int>(optInt(req, "page").value_or(0));
    double wait = optNum(req, "wait").value_or(20.0);
    auto* search = GJSearchObject::create(SearchType::Search, gd::string(query.c_str()));
    search->m_page = page;
    search->retain();
    std::string key = search->getKey();
    auto* glm = GameLevelManager::sharedState();
    glm->getOnlineLevels(search);
    // GameLevelManager stores the result under the search key (or answers
    // from that cache right away); poll it instead of installing a delegate.
    waitUntil(
        [glm, key] { return glm->getStoredOnlineLevels(key.c_str()) != nullptr || !glm->isDLActive(key.c_str()); },
        after(wait), [glm, key, search, query, page, reply](bool) {
            search->release();
            auto* arr = glm->getStoredOnlineLevels(key.c_str());
            if (!arr) {
                reply(errorJson(glm->isDLActive(key.c_str()) ? "timeout" : "search_failed",
                                "no results (server error, no match, or no internet)"));
                return;
            }
            Json out = okJson();
            out["query"] = query;
            out["page"] = page;
            Json levels = Json::array();
            for (auto* l : CCArrayExt<GJGameLevel*>(arr)) levels.push(onlineLevelJson(l));
            out["levels"] = levels;
            reply(out);
        });
}

bool songReady(GJGameLevel* l) {
    if (l->m_songID <= 0) return true;  // built-in track
    return MusicDownloadManager::sharedState()->isSongDownloaded(l->m_songID);
}

void startSong(int songID) {
    if (songID > 0 && !MusicDownloadManager::sharedState()->isSongDownloaded(songID)) {
        MusicDownloadManager::sharedState()->downloadSong(songID);
    }
}

void cmdDownloadLevel(Json const& req, Reply reply) {
    auto id = optInt(req, "id");
    if (!id) fail("bad_request", "missing parameter: id");
    bool song = optBool(req, "song").value_or(true);
    double wait = optNum(req, "wait").value_or(60.0);
    bool force = optBool(req, "force").value_or(false);
    auto* glm = GameLevelManager::sharedState();
    int levelID = static_cast<int>(*id);
    auto have = [glm, levelID] {
        auto* l = glm->getSavedLevel(levelID);
        return l && !l->m_levelString.empty();
    };
    if (force || !have()) glm->downloadLevel(levelID, false, 0);
    waitUntil(have, after(wait), [glm, levelID, song, wait, reply](bool ok) {
        if (!ok) {
            reply(errorJson("timeout", "level " + std::to_string(levelID) +
                                           " did not download (wrong id, server or internet?)"));
            return;
        }
        auto* level = glm->getSavedLevel(levelID);
        auto finish = [level, reply](bool songOk) {
            Json out = okJson();
            out["level"] = onlineLevelJson(level);
            out["song_ready"] = songOk;
            reply(out);
        };
        if (!song || songReady(level)) {
            finish(songReady(level));
            return;
        }
        level->retain();
        startSong(level->m_songID);
        waitUntil([level] { return songReady(level); }, after(wait), [level, finish](bool songOk) {
            finish(songOk);
            level->release();
        });
    });
}

void cmdDownloadSong(Json const& req, Reply reply) {
    auto id = optInt(req, "id");
    if (!id || *id <= 0) fail("bad_request", "id must be a song id > 0");
    int songID = static_cast<int>(*id);
    double wait = optNum(req, "wait").value_or(60.0);
    startSong(songID);
    waitUntil([songID] { return MusicDownloadManager::sharedState()->isSongDownloaded(songID); }, after(wait),
              [songID, reply](bool ok) {
                  if (!ok) {
                      reply(errorJson("timeout", "song " + std::to_string(songID) + " did not download"));
                      return;
                  }
                  Json out = okJson();
                  out["song_id"] = songID;
                  out["path"] = toStd(MusicDownloadManager::sharedState()->pathForSong(songID));
                  reply(out);
              });
}

void cmdCopyToLocal(Json const& req, Reply reply) {
    auto id = optInt(req, "id");
    if (!id) fail("bad_request", "missing parameter: id (a downloaded online level)");
    auto name = reqStr(req, "name");
    bool replace = optBool(req, "replace").value_or(false);
    if (name.empty() || name.size() > 20) fail("bad_request", "name must be 1 to 20 characters (GD's limit)");
    auto* glm = GameLevelManager::sharedState();
    auto* src = glm->getSavedLevel(static_cast<int>(*id));
    if (!src || src->m_levelString.empty()) fail("not_downloaded", "download the level first (download_level)");
    for (auto* l : localLevels()) {
        if (toStd(l->m_levelName) != name) continue;
        if (!replace) fail("exists", "a local level named " + name + " exists (pass replace=true)");
        if (currentPlayLayer() && currentPlayLayer()->m_level == l) fail("in_use", "exit the level first");
        glm->deleteLevel(l);
    }
    // What LevelInfoLayer's clone does: a new editor level with the info copied.
    auto* level = glm->createNewLevel();
    if (!level) fail("create_failed", "createNewLevel returned null");
    int mid = level->m_M_ID;
    level->copyLevelInfo(src);
    level->m_levelName = gd::string(name.c_str());
    if (level->m_levelString.empty()) level->m_levelString = src->m_levelString;
    level->m_levelType = GJLevelType::Editor;
    level->m_levelID = 0;
    level->m_M_ID = mid;
    level->m_originalLevel = static_cast<int>(*id);
    level->m_isEditable = true;
    if (optBool(req, "strip_song").value_or(false)) {
        level->m_songID = 0;
        level->m_audioTrack = 0;
        level->m_songIDs = "";
    }
    if (optBool(req, "save").value_or(true)) LocalLevelManager::sharedState()->save();
    Json out = okJson();
    out["level"] = levelJson(level);
    out["source"] = onlineLevelJson(src);
    reply(out);
}

void cmdGetLevelString(Json const& req, Reply reply) {
    auto* level = resolveLevel(req);
    Json out = okJson();
    out["level"] = levelJson(level);
    out["level_string"] = toStd(level->m_levelString);  // as stored: usually gzip+base64 ("H4sI...")
    reply(out);
}

void cmdSetLevelString(Json const& req, Reply reply) {
    Json r = req;
    r["type"] = "local";
    auto* level = resolveLevel(r);
    auto name = toStd(level->m_levelName);
    auto prefix = optStr(req, "prefix").value_or("gdrl-");
    if (name.rfind(prefix, 0) != 0 && !optBool(req, "force").value_or(false)) {
        fail("refused", "only local levels named " + prefix + "* are edited without force=true");
    }
    if (currentPlayLayer() && currentPlayLayer()->m_level == level) fail("in_use", "exit the level first");
    if (currentEditor()) fail("in_editor", "close the editor first");
    auto data = reqStr(req, "level_string");
    gd::string gs(data.c_str());
    if (data.rfind("H4sI", 0) != 0) gs = ZipUtils::compressString(gs, false, 0);
    level->m_levelString = gs;
    if (optBool(req, "save").value_or(true)) LocalLevelManager::sharedState()->save();
    Json out = okJson();
    out["level"] = levelJson(level);
    reply(out);
}

void cmdScene(Json const&, Reply reply) { reply(sceneJson()); }

using Handler = void (*)(Json const&, Reply);

const std::map<std::string, Handler>& handlers() {
    static const std::map<std::string, Handler> table = {
        {"ping", cmdPing},
        {"scene", cmdScene},
        {"tree", cmdTree},
        {"click", cmdClick},
        {"key", cmdKey},
        {"screenshot", cmdScreenshot},
        {"open_level", cmdOpenLevel},
        {"exit_level", cmdExitLevel},
        {"menu", cmdMenu},
        {"close_popups", cmdClosePopups},
        {"list_levels", cmdListLevels},
        {"create_level", cmdCreateLevel},
        {"delete_level", cmdDeleteLevel},
        {"set_mod_enabled", cmdSetModEnabled},
        {"set_mod_setting", cmdSetModSetting},
        {"quit_game", cmdQuit},
        {"prevent_pause", cmdPreventPause},
        {"get_option", cmdGetOption},
        {"set_option", cmdSetOption},
        {"search_levels", cmdSearchLevels},
        {"download_level", cmdDownloadLevel},
        {"download_song", cmdDownloadSong},
        {"copy_to_local", cmdCopyToLocal},
        {"get_level_string", cmdGetLevelString},
        {"set_level_string", cmdSetLevelString},
    };
    return table;
}

}  // namespace

State& state() {
    static State s;
    return s;
}

bool preventPause() {
    if (state().preventPauseOverride) return *state().preventPauseOverride;
    return Mod::get()->getSettingValue<bool>("prevent-pause");
}

void dispatch(Json const& req, Reply reply) {
    std::string cmd;
    if (req.contains("cmd")) {
        if (auto c = req["cmd"].asString()) cmd = c.unwrap();
    }
    auto it = handlers().find(cmd);
    if (it == handlers().end()) {
        reply(errorJson("unknown_command", "unknown cmd: " + cmd));
        return;
    }
    // A reply that may only fire once, even if a handler errors after replying.
    auto fired = std::make_shared<bool>(false);
    Reply once = [fired, reply](Json v) {
        if (*fired) return;
        *fired = true;
        reply(std::move(v));
    };
    try {
        it->second(req, once);
    } catch (CmdError const& e) {
        once(errorJson(e.code, e.message));
    } catch (std::exception const& e) {
        once(errorJson("exception", e.what()));
    }
}

}  // namespace gdctl
