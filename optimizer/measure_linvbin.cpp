#include <vector>
#include <stdlib.h>
#include <fstream>
#include <stdint.h>
#include <chrono>
#include <iostream>
#include <algorithm>

#define DATA_SIZE 200000000

int linear_search(std::vector<uint64_t>& data, uint64_t start, uint64_t key) {
  if (data[start] == key) return start;
  if (data[start] < key) {
    // forward search
    while (data[++start] != key);
    return start;
  } else {
    // backwards search
    while (data[--start] != key);
    return start;
  }
}


int main() {
  srand(42);

  std::vector<uint64_t> data_pts;
  std::vector<uint64_t> search_keys;
  
  for (unsigned int i = 0; i < DATA_SIZE; i++) {
    data_pts.push_back(i);
  }

  for (unsigned int i = 0; i < 50000; i++) {
    search_keys.push_back(rand() % DATA_SIZE);
  }

  for (unsigned int err = 1; err < DATA_SIZE / 2; err *= 2) {
    std::vector<uint64_t> starting_points;
    for (auto key : search_keys) {
      if (key < err) {
        starting_points.push_back(key + err);
      } else if (key + err >= DATA_SIZE) {
        starting_points.push_back(key - err);
      } else {
        if (rand() % 2 == 0) {
          starting_points.push_back(key + err);
        } else {
          starting_points.push_back(key - err);
        }
      }
    }

    if (err < 5000) {
      auto start = std::chrono::high_resolution_clock::now();
      __attribute__((unused)) volatile uint64_t pos;
      for (unsigned int i = 0; i < search_keys.size(); i++) {
        uint64_t key = search_keys[i];
        uint64_t start_at = starting_points[i];
        pos = linear_search(data_pts, start_at, key);
      }
      auto finish = std::chrono::high_resolution_clock::now();
      std::cout << "linear," << err << ","
                << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "\n";
    }

    {
      auto start = std::chrono::high_resolution_clock::now();
      __attribute__((unused)) volatile uint64_t pos;
      for (unsigned int i = 0; i < search_keys.size(); i++) {
        uint64_t key = search_keys[i];
        uint64_t start_at = starting_points[i];
        uint64_t start = (start_at >= err ? start_at - err : 0);
        uint64_t end = (start_at + err > DATA_SIZE ? DATA_SIZE : start_at + err);

        pos = std::binary_search(data_pts.begin() + start, data_pts.begin() + end, key);
      }
      auto finish = std::chrono::high_resolution_clock::now();
      std::cout << "binary," << err << ","
                << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "\n";
    }
  }

  
    
}
