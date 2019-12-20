#include "benchmark.h"
#include "util.h"
#include "utils/cxxopts.hpp"

#include "competitors/rmi_search.h"

using namespace std;


#define NAME2(a, b)         NAME2_HIDDEN(a,b)
#define NAME2_HIDDEN(a, b)  a ## b

#define NAME3(a, b, c)         NAME3_HIDDEN(a,b,c)
#define NAME3_HIDDEN(a, b, c)  a ## b ## c

#define NAME5(a, b, c, d, e)         NAME5_HIDDEN(a,b,c,d,e)
#define NAME5_HIDDEN(a, b, c, d, e)  a ## b ## c ## d ## e

#define run_rmi_linear(dtype, name, suffix)  if (filename.find("/" #name "_" #dtype) != std::string::npos) { benchmark.Run<RMI_L<NAME2(dtype, _t), NAME5(name, _, dtype, _, suffix)::BUILD_TIME_NS, NAME5(name, _, dtype, _,suffix)::RMI_SIZE, NAME5(name, _, dtype, _,suffix)::NAME, NAME5(name, _, dtype, _, suffix)::lookup>>(); }

#define run_rmi_binary(dtype, name, suffix)  if (filename.find("/" #name "_" #dtype) != std::string::npos) { benchmark.Run<RMI_B<NAME2(dtype, _t), NAME5(name, _, dtype, _, suffix)::BUILD_TIME_NS, NAME5(name, _, dtype, _, suffix)::RMI_SIZE, NAME5(name, _, dtype, _,suffix)::NAME, NAME5(name, _, dtype, _, suffix)::lookup>>(); }

int main(int argc, char* argv[]) {
  cxxopts::Options options("benchmark", "Searching on sorted data benchmark");
  options.positional_help("<data> <lookups>");
  options.add_options()
      ("data", "Data file with keys", cxxopts::value<std::string>())
      ("lookups", "Lookup key (query) file", cxxopts::value<std::string>())
      ("help", "Displays help")
      ("r,repeats",
       "Number of repeats",
       cxxopts::value<int>()->default_value("1"))
      ("p,perf", "Track performance counters")
      ("b,build", "Only measure and report build times")
      ("histogram", "Measure each lookup and output histogram data")
      ("positional",
       "extra positional arguments",
       cxxopts::value<std::vector<std::string>>());

  options.parse_positional({"data", "lookups", "positional"});

  const auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help({}) << "\n";
    exit(0);
  }

  const size_t num_repeats = result["repeats"].as<int>();
  cout << "Repeating lookup code " << num_repeats << " time(s)." << endl;

  const bool perf = result.count("perf");
  const bool build = result.count("build");
  const bool histogram = result.count("histogram");
  const std::string filename = result["data"].as<std::string>();
  const std::string lookups = result["lookups"].as<std::string>();

  const DataType type = util::resolve_type(filename);

  if (lookups.find("lookups")==std::string::npos) {
    cerr
        << "Warning: lookups file seems misnamed. Did you specify the right one?\n";
  }

  // Pin main thread to core 0.
  util::set_cpu_affinity(0);

  switch (type) {
    case DataType::UINT32: {
      break;
    }
    case DataType::UINT64: {
      // Create benchmark.
      sosd::Benchmark<uint64_t>
          benchmark(filename, lookups, num_repeats, perf, build, histogram);

      
      // RMIs
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm0::NAME, nm0::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm8::NAME, nm8::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm1::NAME, nm1::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm9::NAME, nm9::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm17::NAME, nm17::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm25::NAME, nm25::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm33::NAME, nm33::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm10::NAME, nm10::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm18::NAME, nm18::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm3::NAME, nm3::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm11::NAME, nm11::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm19::NAME, nm19::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm59::NAME, nm59::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm4::NAME, nm4::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm12::NAME, nm12::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm20::NAME, nm20::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm36::NAME, nm36::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm52::NAME, nm52::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm60::NAME, nm60::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm5::NAME, nm5::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm21::NAME, nm21::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm37::NAME, nm37::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm6::NAME, nm6::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm14::NAME, nm14::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm22::NAME, nm22::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm30::NAME, nm30::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm38::NAME, nm38::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm46::NAME, nm46::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm54::NAME, nm54::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm31::NAME, nm31::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm55::NAME, nm55::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm70::NAME, nm70::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm86::NAME, nm86::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm102::NAME, nm102::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm110::NAME, nm110::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm134::NAME, nm134::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm150::NAME, nm150::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm158::NAME, nm158::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm166::NAME, nm166::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm174::NAME, nm174::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm190::NAME, nm190::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm71::NAME, nm71::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm87::NAME, nm87::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm95::NAME, nm95::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm103::NAME, nm103::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm111::NAME, nm111::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm135::NAME, nm135::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm143::NAME, nm143::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm151::NAME, nm151::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm159::NAME, nm159::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm167::NAME, nm167::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm72::NAME, nm72::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm80::NAME, nm80::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm96::NAME, nm96::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm112::NAME, nm112::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm144::NAME, nm144::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm160::NAME, nm160::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm168::NAME, nm168::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm176::NAME, nm176::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm184::NAME, nm184::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm73::NAME, nm73::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm81::NAME, nm81::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm97::NAME, nm97::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm105::NAME, nm105::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm121::NAME, nm121::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm137::NAME, nm137::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm161::NAME, nm161::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm177::NAME, nm177::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm185::NAME, nm185::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm82::NAME, nm82::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm98::NAME, nm98::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm106::NAME, nm106::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm114::NAME, nm114::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm122::NAME, nm122::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm146::NAME, nm146::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm154::NAME, nm154::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm162::NAME, nm162::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm178::NAME, nm178::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm186::NAME, nm186::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm83::NAME, nm83::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm107::NAME, nm107::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm115::NAME, nm115::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm139::NAME, nm139::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm147::NAME, nm147::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm179::NAME, nm179::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm108::NAME, nm108::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm116::NAME, nm116::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm124::NAME, nm124::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm132::NAME, nm132::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm156::NAME, nm156::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm180::NAME, nm180::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm188::NAME, nm188::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm85::NAME, nm85::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm101::NAME, nm101::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm109::NAME, nm109::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm125::NAME, nm125::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm133::NAME, nm133::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm165::NAME, nm165::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm181::NAME, nm181::lookup>>();
      benchmark.Run<RMI_B<uint64_t, 0, 0, nm189::NAME, nm189::lookup>>();

      break;
    }
  }

  return 0;
}
