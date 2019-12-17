#define FILENAME "{{filename}}"

{{#namespaces}}
#include "opt/{{.}}.h"
{{/namespaces}}
#include <vector>
#include <stdlib.h>
#include <fstream>
#include <stdint.h>
#include <chrono>
#include <iostream>

int main() {
  srand(42);
  std::vector<uint64_t> data;
  std::ifstream in(FILENAME, std::ios::binary);

  // Read size.
  uint64_t size;
  in.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
  data.resize(size);
  // Read values.
  in.read(reinterpret_cast<char*>(data.data()), size*sizeof(uint64_t));
  in.close();

  std::vector<uint64_t> random_keys;
    
  for (unsigned int i = 0; i < 100000; i++) {
    unsigned int idx = rand() % data.size();
    random_keys.push_back(data[idx]);
  }

  {{#namespaces}}
  {
    auto start = std::chrono::high_resolution_clock::now();
    __attribute__((unused)) volatile uint64_t pos;
    for (auto itm : random_keys) {
      pos = {{.}}::lookup(itm);
    }
    
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";
  }
  {{/namespaces}}
}
