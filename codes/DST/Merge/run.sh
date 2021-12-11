python ../CatChoice/make_data.py -d $1/test_seen -s $1/schema.json -o cat_task_data/test_seen.jsonl --norm && \
python ../CatChoice/predict.py --test_file cat_task_data/test_seen.jsonl --target_dir ../models/cat_choice/ --out_file outputs/test_seen/cat_results.json
python ../NoncatSpan/make_data.py -d $1/test_seen -s $1/schema.json -o noncat_task_data/test_seen.jsonl --norm && \
python ../NoncatSpan/predict.py --test_file noncat_task_data/test_seen.jsonl --target_dir ../models/noncat_span/ --out_file outputs/test_seen/noncat_results.json
python merge_to_csv.py --in_files outputs/test_seen/cat_results.json outputs/test_seen/noncat_results.json --out_csv outputs/test_seen/results.csv

python ../CatChoice_WD/make_data.py -d $1/test_unseen -s $1/schema.json -o cat_task_data/test_unseen.jsonl --norm && \
python ../CatChoice_WD/predict.py --test_file cat_task_data/test_unseen.jsonl --target_dir ../models/cat_choice_wd/ --out_file outputs/test_unseen/cat_results.json
python ../NoncatSpan/make_data.py -d $1/test_unseen -s $1/schema.json -o noncat_task_data/test_unseen.jsonl --norm && \
python ../NoncatSpan/predict.py --test_file noncat_task_data/test_unseen.jsonl --target_dir ../models/noncat_span/ --out_file outputs/test_unseen/noncat_results.json
python merge_to_csv.py --in_files outputs/test_unseen/cat_results.json outputs/test_unseen/noncat_results.json --out_csv outputs/test_unseen/results.csv

