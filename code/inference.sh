python inference.py \
--output_dir "./outputs/test_dataset/" \
--dataset_name "../data/test_dataset/" \
--model_name_or_path "./models/train_dataset/" \
--do_predict \
--overwrite_output_dir True \
--retrieval_method bm25 \
--eval_retrieval True \