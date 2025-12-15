# python /data100t1/home/biand/Projects/PheWES/src/phecode_enrichment_with_permutation.py \
# --control_fn /data100t1/home/biand/Projects/PheWES/results/an_1203/case_control_pairs_train.txt \
# --output_path /data100t1/home/biand/Projects/PheWES/results/an_1203 \
# --output_prefix an_1203 \
# --phecode_binary_feather_file /data100t1/home/biand/Projects/PheWES/results/an_1203/binary_phecode.feather \
# --n_permute 100000

python /data100t1/home/biand/Projects/PheWES/src/phecode_enrichment_generate_reports.py \
--output_folder /data100t1/home/biand/Projects/PheWES/results/an_1203 \
--trait an \
--input_prefix an_1203