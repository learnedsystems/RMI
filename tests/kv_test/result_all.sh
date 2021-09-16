set -e

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

mkdir -p out

echo "Testing mmap"
for ((i = 0; i < ${#DATASET_NAMES[@]}; i++)) do
  for ((j = 0; j < 5; j++)) do
    dataset_name="${DATASET_NAMES[$i]}"
    echo ">>> ${dataset_name} ${j}"
    bash ~/reload_nfs.sh
    ./main_${dataset_name}_rmi_mmap --data_path=data/${dataset_name} --key_path=data/${dataset_name}_keyset --rmi_data_path=storage/${dataset_name}_rmi_mmap/rmi_data --out_path=out/out_main_${dataset_name}_rmi_mmap.txt 2>& 1 | tee log.txt
  done
done

echo "Testing whole"
for ((i = 0; i < ${#DATASET_NAMES[@]}; i++)) do
  for ((j = 0; j < 5; j++)) do
    dataset_name="${DATASET_NAMES[$i]}"
    echo ">>> ${dataset_name} ${j}"
    bash ~/reload_nfs.sh
    ./main_${dataset_name}_rmi_whole --data_path=data/${dataset_name} --key_path=data/${dataset_name}_keyset --rmi_data_path=storage/${dataset_name}_rmi_whole/rmi_data --out_path=out/out_main_${dataset_name}_rmi_whole.txt 2>& 1 | tee log.txt
  done
done