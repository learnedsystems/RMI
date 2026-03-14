#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "rmi.h"

class RMILearnedIndex {
 public:
  bool Load(const std::string& root,
            const std::string& dataset_namespace = rmi::NAME) {
    std::filesystem::path full_path = std::filesystem::path(root) / dataset_namespace;
    if (!std::filesystem::exists(full_path)) {
      full_path = std::filesystem::path(root);
    }
    data_path_ = full_path.string();
    loaded_ = rmi::load(data_path_.c_str());
    return loaded_;
  }

  uint64_t Lookup(uint64_t key, size_t* err) const {
    return rmi::lookup(key, err);
  }

  void Cleanup() {
    rmi::cleanup();
    loaded_ = false;
  }

  size_t SizeBytes() const { return rmi::RMI_SIZE; }
  uint64_t BuildTimeNs() const { return rmi::BUILD_TIME_NS; }

 private:
  std::string data_path_;
  bool loaded_{false};
};
