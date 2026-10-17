// Checks that the bound returned by lookup contains the lower bound of the
// key, mostly for keys outside the range of the data: every key, key - 1,
// key + 1, key + 2^32, 0, 2^64 - 1, and evenly spaced keys over the whole
// key space and between the smallest and the largest key.
//
// Run as ./test <case>; see the Makefile for the cases.

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>
#include "rmi_cubic_line.h"
#include "rmi_cubic_turn.h"
#include "rmi_dense.h"
#include "rmi_dense32.h"
#include "rmi_radix_marker.h"
#include "rmi_radix_oob.h"
#include "rmi_radix_prefix.h"
#include "rmi_radix32.h"

template <class K>
static std::vector<K> load_keys(const char* fn) {
  std::ifstream in(fn, std::ios::binary);
  uint64_t size;
  in.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
  std::vector<K> data(size);
  in.read(reinterpret_cast<char*>(data.data()), size * sizeof(K));
  return data;
}

// K is the key type of the data file; lookup takes 64-bit keys either way.
template <class K>
static int check(const char* data_file,
                 bool (*load)(char const*),
                 uint64_t (*lookup)(uint64_t, size_t*),
                 void (*cleanup)()) {
  const std::vector<K> raw = load_keys<K>(data_file);
  const std::vector<uint64_t> data(raw.begin(), raw.end());
  if (data.empty() || !load("rmi_data")) {
    std::cout << "Could not load the data or the RMI." << std::endl;
    return 1;
  }

  const uint64_t max_value = std::numeric_limits<uint64_t>::max();
  std::vector<uint64_t> queries = {0, max_value};
  for (uint64_t key : data) {
    queries.push_back(key);
    if (key > 0) queries.push_back(key - 1);
    if (key < max_value) queries.push_back(key + 1);
    if (key <= max_value - (1ULL << 32)) queries.push_back(key + (1ULL << 32));
  }
  // evenly spaced over the whole key space (the step is not a power of two,
  // so the low bits vary too) and between the smallest and the largest key
  const uint64_t min_key = data.front(), max_key = data.back();
  for (uint64_t i = 0; i < 65536; i++) {
    queries.push_back(max_value / 65537 * i);
    queries.push_back(min_key + (max_key - min_key) / 65536 * i);
  }

  uint64_t num_failed = 0;
  for (uint64_t key : queries) {
    const uint64_t true_index = (uint64_t)
      std::distance(data.begin(), std::lower_bound(data.begin(), data.end(), key));
    size_t err;
    const uint64_t guess = lookup(key, &err);
    const uint64_t diff = (guess > true_index ? guess - true_index : true_index - guess);
    if (diff > err) {
      if (num_failed < 5) {
        std::cout << "Search key: " << key
                  << " lower bound: " << true_index
                  << " RMI guess: " << guess << " +/- " << err
                  << " diff: " << diff << std::endl;
      }
      num_failed++;
    }
  }

  cleanup();
  std::cout << num_failed << " of " << queries.size() << " lookups failed" << std::endl;
  return num_failed == 0 ? 0 : 1;
}

#define CASE(name, key_type, data_file, ns) \
  if (strcmp(argv[1], name) == 0) { \
    std::cout << name << std::endl; \
    return check<key_type>(data_file, ns::load, ns::lookup, ns::cleanup); \
  }

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "usage: " << argv[0] << " <case>" << std::endl;
    return 1;
  }
  CASE("dense", uint64_t, "data_dense_uint64", rmi_dense);
  CASE("dense32", uint32_t, "data_dense_uint32", rmi_dense32);
  CASE("cubic_line", uint64_t, "data_line_uint64", rmi_cubic_line);
  CASE("cubic_turn", uint64_t, "data_turn_uint64", rmi_cubic_turn);
  CASE("radix_oob", uint64_t, "data_radix_uint64", rmi_radix_oob);
  CASE("radix_marker", uint64_t, "data_radix_uint64", rmi_radix_marker);
  CASE("radix_prefix", uint64_t, "data_prefix_uint64", rmi_radix_prefix);
  CASE("radix32", uint32_t, "data_split_uint32", rmi_radix32);
  std::cout << "unknown case " << argv[1] << std::endl;
  return 1;
}
