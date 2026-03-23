#pragma once
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/file.h>
#include <unistd.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>

namespace rj = rapidjson;

inline double get_double(const rj::Value& obj, const char* key, double default_val) {
    if (obj.HasMember(key) && obj[key].IsNumber()) return obj[key].GetDouble();
    return default_val;
}

inline int get_int(const rj::Value& obj, const char* key, int default_val) {
    if (obj.HasMember(key) && obj[key].IsInt()) return obj[key].GetInt();
    return default_val;
}

inline uint64_t get_uint64(const rj::Value& obj, const char* key, uint64_t default_val) {
    if (obj.HasMember(key) && obj[key].IsUint64()) return obj[key].GetUint64();
    return default_val;
}

inline bool get_bool(const rj::Value& obj, const char* key, bool default_val) {
    if (obj.HasMember(key) && obj[key].IsBool()) return obj[key].GetBool();
    return default_val;
}

inline std::string get_string(const rj::Value& obj, const char* key, const std::string& default_val) {
    if (obj.HasMember(key) && obj[key].IsString()) return obj[key].GetString();
    return default_val;
}

inline std::string hash_config(const std::string& content) {
    size_t h = std::hash<std::string>{}(content);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << h;
    return oss.str();
}

inline std::string read_file(const std::string& path) {
    std::ifstream f(path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline void update_registry(const std::string& registry_path, const std::string& hash,
                            const std::string& config_content, uint64_t seed_used = 0) {
    // File lock for concurrent safety
    std::string lock_path = registry_path + ".lock";
    int lock_fd = open(lock_path.c_str(), O_CREAT | O_RDWR, 0666);
    if (lock_fd >= 0) flock(lock_fd, LOCK_EX);

    rj::Document registry;
    registry.SetObject();

    std::ifstream ifs(registry_path);
    if (ifs.is_open()) {
        rj::IStreamWrapper isw(ifs);
        registry.ParseStream(isw);
        ifs.close();
        if (registry.HasParseError() || !registry.IsObject()) {
            registry.SetObject();
        }
    }

    rj::Document config_doc;
    config_doc.Parse(config_content.c_str());

    rj::Value entry(rj::kObjectType);
    auto& alloc = registry.GetAllocator();

    rj::Value config_copy;
    config_copy.CopyFrom(config_doc, alloc);
    entry.AddMember("config", config_copy, alloc);
    if (seed_used != 0) {
        entry.AddMember("seed_used", seed_used, alloc);
    }

    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ts;
    ts << std::put_time(std::localtime(&t), "%Y-%m-%dT%H:%M:%S");
    entry.AddMember("timestamp", rj::Value(ts.str().c_str(), alloc), alloc);

    rj::Value key(hash.c_str(), alloc);
    if (registry.HasMember(hash.c_str())) {
        registry[hash.c_str()] = entry;
    } else {
        registry.AddMember(key, entry, alloc);
    }

    std::ofstream ofs(registry_path);
    rj::OStreamWrapper osw(ofs);
    rj::PrettyWriter<rj::OStreamWrapper> writer(osw);
    registry.Accept(writer);
    ofs << "\n";

    if (lock_fd >= 0) {
        flock(lock_fd, LOCK_UN);
        close(lock_fd);
    }
}
