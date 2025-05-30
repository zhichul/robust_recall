for i in $(seq 0 9)
do
for range in 0_to_1000 1000_to_10000 10000_to_100000 100000_to_inf
do
python3 preprocess_data.py \
    --local_dir data/processed/split${i}/${range} \
    --data_source data/raw/splits/fifty_fifty/split${i}/${range} \
    --data_name $range
done
done