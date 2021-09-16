set -e
echo "Training with compiler flags ${CXX} ${CXXFLAGS} ${CXXBACKFLAGS}"

DATASET_NAMES=(
  "books_800M_uint64"
  "fb_200M_uint64"
  "lognormal_200M_uint64"
  "normal_200M_uint64"
  "osm_cellids_800M_uint64"
  "uniform_dense_200M_uint64"
  "uniform_sparse_200M_uint64"
  "wiki_ts_200M_uint64"
 )
# RMI_CONFIGS=( # configurations obtained from rmi optimizer
#   "linear_spline,linear      32768" # books_800M_uint64
#   "robust_linear,linear      32768" # fb_200M_uint64
#   "robust_linear,linear 32768" # lognormal_200M_uint64
#   "robust_linear,linear 32768" # normal_200M_uint64
#   "linear,linear             32768" # osm_cellids_800M_uint64
#   "linear_spline,linear      32768" # uniform_dense_200M_uint64
#   "linear_spline,linear      32768" # uniform_sparse_200M_uint64
#   "linear,linear             32768" # wiki_ts_200M_uint64
# )
RMI_CONFIGS=( # configurations obtained from rmi optimizer
  "linear_spline,linear   16777216" # books_800M_uint64
  "robust_linear,linear   16777216" # fb_200M_uint64
  "robust_linear,linear 16777216" # lognormal_200M_uint64
  "robust_linear,linear 16777216" # normal_200M_uint64
  "cubic,linear           16777216" # osm_cellids_800M_uint64
  "radix,linear_spline     8388608" # uniform_dense_200M_uint64
  "linear_spline,linear   16777216" # uniform_sparse_200M_uint64
  "linear_spline,linear   16777216" # wiki_ts_200M_uint64
)

echo "Training mmap"
for ((i = 0; i < ${#DATASET_NAMES[@]}; i++)) do
  dataset_name="${DATASET_NAMES[$i]}"
  rmi_config="${RMI_CONFIGS[$i]}"
  echo ">>> ${dataset_name}, config= ${rmi_config}"
  ../rmi data/${dataset_name} rmi ${rmi_config} --use-mmap  # TODO: tune
  rm -rf storage/${dataset_name}_rmi_mmap
  mkdir storage/${dataset_name}_rmi_mmap
  mv rmi.cpp rmi.h rmi_data.h rmi_data storage/${dataset_name}_rmi_mmap

  ${CXX} ${CXXFLAGS} main.cpp storage/${dataset_name}_rmi_mmap/rmi.cpp -I storage/${dataset_name}_rmi_mmap -I . -o main_${dataset_name}_rmi_mmap ${CXXBACKFLAGS}
done

echo "Training whole"
for ((i = 0; i < ${#DATASET_NAMES[@]}; i++)) do
  dataset_name="${DATASET_NAMES[$i]}"
  rmi_config="${RMI_CONFIGS[$i]}"
  echo ">>> ${dataset_name}, config= ${rmi_config}"
  rm -rf storage/${dataset_name}_rmi_whole
  mkdir storage/${dataset_name}_rmi_whole
  ../rmi data/${dataset_name} rmi ${rmi_config}  # TODO: tune
  mv rmi.cpp rmi.h rmi_data.h rmi_data storage/${dataset_name}_rmi_whole

  ${CXX} ${CXXFLAGS} main.cpp storage/${dataset_name}_rmi_whole/rmi.cpp -I storage/${dataset_name}_rmi_whole -I . -o main_${dataset_name}_rmi_whole ${CXXBACKFLAGS}
done