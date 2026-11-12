// NeuralRysum real-time engine - model file (.nrys) loader.
// Header-only, no dependencies beyond the C++17 standard library.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nrys {

struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;
};

// Parsed .nrys file. Loading allocates; do it off the audio thread.
class ModelFile {
public:
    bool loadFromMemory(const void* bytes, size_t size, std::string& err) {
        try {
            parse(static_cast<const char*>(bytes), size);
            return true;
        } catch (const std::exception& e) {
            err = e.what();
            cfg.clear();
            tensors.clear();
            return false;
        }
    }

    bool load(const std::string& path, std::string& err) {
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            err = "cannot open " + path;
            return false;
        }
        std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        return loadFromMemory(bytes.data(), bytes.size(), err);
    }

    const Tensor& t(const std::string& name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("model file has no tensor '" + name + "'");
        return it->second;
    }

    const std::vector<int>& list(const std::string& key) const {
        auto it = cfg.find(key);
        if (it == cfg.end() || it->second.empty())
            throw std::runtime_error("model file has no config entry '" + key + "'");
        return it->second;
    }

    int num(const std::string& key) const { return list(key)[0]; }

    std::map<std::string, std::vector<int>> cfg;
    std::map<std::string, Tensor> tensors;

private:
    void parse(const char* p, size_t size) {
        if (size < 12 || std::memcmp(p, "NRYS0001", 8) != 0)
            throw std::runtime_error("not a NeuralRysum model file");
        uint32_t hlen = 0;
        std::memcpy(&hlen, p + 8, 4);
        if (12 + static_cast<size_t>(hlen) > size) throw std::runtime_error("truncated header");
        std::istringstream header(std::string(p + 12, hlen));
        const char* data = p + 12 + hlen;
        const size_t nFloats = (size - 12 - hlen) / 4;

        std::string line;
        while (std::getline(header, line)) {
            std::istringstream ls(line);
            std::string kind, name;
            ls >> kind >> name;
            if (kind == "cfg") {
                int v;
                while (ls >> v) cfg[name].push_back(v);
            } else if (kind == "tensor") {
                size_t offset = 0, count = 1;
                int ndim = 0;
                ls >> offset >> ndim;
                Tensor t;
                for (int i = 0; i < ndim; ++i) {
                    int d = 0;
                    ls >> d;
                    if (d <= 0) throw std::runtime_error("bad shape for " + name);
                    t.shape.push_back(d);
                    count *= static_cast<size_t>(d);
                }
                if (ls.fail() || offset + count > nFloats)
                    throw std::runtime_error("truncated data for " + name);
                t.data.resize(count);
                std::memcpy(t.data.data(), data + offset * 4, count * 4);
                tensors[name] = std::move(t);
            }
        }
    }
};

}  // namespace nrys
