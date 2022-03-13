# bash optimizer.sh ../data ./output

DATASET=$1
OUTPUT_PATH=$2

echo "Auto-tuning optimizer starts"
echo "Dataset path: ${DATASET}"
echo "Output path: ${OUTPUT_PATH}"

for file in ${DATASET}/*
do
    FILE_NAME="${file##*/}" # Extract file name from path
    echo ">> Auto-tuning dataset: ${FILE_NAME}"
    OUTPUT_FILE="${OUTPUT_PATH}/${FILE_NAME}.json"
    if [ -f "$OUTPUT_FILE" ]; then
	echo "${OUTPUT_FILE} already exists"
    else
	echo "Optimizer starts"
	cargo run --release -- --optimize ${OUTPUT_FILE} ${file}
    fi
done

