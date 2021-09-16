#include <algorithm>
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "flags.h"
#include "mmap_struct.h"
#include "rmi.h"

double report_t(size_t t_idx, std::chrono::time_point<std::chrono::high_resolution_clock> start_t) {
  auto lookups_end_time = std::chrono::high_resolution_clock::now();
  auto lookup_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      lookups_end_time - start_t)
                      .count();
  std::cout << "<<< " << lookup_time << " ns " << " to finish " << (t_idx + 1) << " queries." << std::endl;
  return std::chrono::duration<double>(lookup_time).count();
}


int main(int argc, char* argv[]) {
  // load flags
  auto flags = parse_flags(argc, argv);

  // extract paths
  std::string data_path = get_required(flags, "data_path");
  std::string key_path = get_required(flags, "key_path");
  std::string rmi_data_path = get_required(flags, "rmi_data_path");
  std::string out_path = get_required(flags, "out_path");

  // TODO: load keys
  auto queries = std::vector<uint64_t>();
  std::ifstream query_words_in(key_path);
  std::string line;
  while (std::getline(query_words_in, line)) {
    std::istringstream input;
    input.str(line);

    std::string key;
    input >> key;

    queries.push_back(std::stoull(key));
  }

  // start timer
  auto start_t = std::chrono::high_resolution_clock::now();

  // Load the data
  KeyArray<uint64_t> data(data_path.c_str());
  // report_t(777, start_t);

  // Load RMI
  std::cout << "RMI status: " << rmi::load(rmi_data_path.c_str()) << std::endl;
  // report_t(888, start_t);

  // for (uint64_t key_index = 0; key_index < data.size(); key_index++) {
  size_t err;
  auto timestamps = std::vector<double>();
  for (size_t t_idx = 0; t_idx < queries.size(); t_idx++) {
    // TODO: sample this
    uint64_t lookup = queries[t_idx];

    // rmi index
    size_t rmi_guess = (size_t) rmi::lookup(lookup, &err);

    // error correction
    size_t guess_left = (rmi_guess >= err) ? rmi_guess - err : 0;
    size_t guess_right = rmi_guess + err;
    size_t true_index = data.rank_within(lookup, guess_left, guess_right);
    
    if (((((t_idx + 1) & (t_idx)) == 0) && (t_idx < 999)) || ((t_idx >= 999) & (t_idx % 1000 == 999))) {
      timestamps.push_back(report_t(t_idx, start_t));
    }

    if (t_idx % 10000 == 0) { // UNCOMMENT to debug
      // compute error, TODO: mute this?
      uint64_t diff = (rmi_guess > true_index ? rmi_guess - true_index : true_index - rmi_guess);

      // print message
      std::cout << "Search key: " << lookup
                << " RMI guess: " << rmi_guess << " +/- " << err
                << " Key at " << true_index << ": " << data[true_index]
                << " diff: " << diff << std::endl;
    }
  }

  // write result to file
  std::ofstream file_out;
  file_out.open(out_path, std::ios_base::app);
  for (const auto& timestamp : timestamps) {
    file_out << timestamp / 1000000.0 << ",";
  }
  file_out << std::endl;
  file_out.close();

  // clean up data
  delete &data;
  
  // clean up index
  rmi::cleanup();

  return 0;
}
