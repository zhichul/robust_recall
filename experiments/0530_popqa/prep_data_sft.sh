for i in 0 #$(seq 0 9)
do
for range in 0_to_1000 1000_to_10000 10000_to_100000 100000_to_inf
do
python3 preprocess_data_sft.py \
    --local_dir data/sft/split${i}/${range} \
    --data_source data/raw/splits/fifty_fifty/split${i}/${range} \
    --data_name $range \
    --decontaminated
done
done