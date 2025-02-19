ROOT_VIDEO="/path/to/video/folder"
ROOT_CLIPS="/path/to/video/clips/folder"
ROOT_META="/path/to/meta/folder"

# # 1.1 Create a meta file from a video folder. This should output ${ROOT_META}/meta.csv
# python -m tools.datasets.convert video ${ROOT_VIDEO} --output ${ROOT_META}/meta.csv

# # 1.2 Get video information and remove broken videos. This should output ${ROOT_META}/meta_info_fmin1.csv
# python -m tools.datasets.datautil ${ROOT_META}/meta.csv --info --fmin 1

# # 2.1 Detect scenes. This should output ${ROOT_META}/meta_info_fmin1_timestamp.csv
# python -m tools.scene_cut.scene_detect ${ROOT_META}/meta_info_fmin1.csv

# # 2.2 Cut video into clips based on scenes. This should produce video clips under ${ROOT_CLIPS}
# python -m tools.scene_cut.cut ${ROOT_META}/meta_info_fmin1_timestamp.csv --save_dir ${ROOT_CLIPS}

python -m tools.scene_cut.cut /data/shanglinyuan/ysl-opensora-data/datasets_csv/VidGen/Subset1/video_info_fmin1.csv --save_dir /data/shanglinyuan/ysl-opensora-data/datasets_csv/VidGen/Test_data3 --no_scene_split --output_fps 15

# # 2.3 Create a meta file for video clips. This should output ${ROOT_META}/meta_clips.csv
# python -m tools.datasets.convert video ${ROOT_CLIPS} --output ${ROOT_META}/meta_clips.csv

# # 2.4 Get clips information and remove broken ones. This should output ${ROOT_META}/meta_clips_info_fmin1.csv
# python -m tools.datasets.datautil ${ROOT_META}/meta_clips.csv --info --fmin 1

# # 3.1 Predict aesthetic scores. This should output ${ROOT_META}/meta_clips_info_fmin1_aes.csv
# torchrun --nproc_per_node 4 -m tools.scoring.aesthetic.inference \
#   ${ROOT_META}/meta_clips_info_fmin1.csv \
#   --bs 1024 \
#   --num_workers 4

# # 3.2 Filter by aesthetic scores. This should output ${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5.csv
# python -m tools.datasets.datautil ${ROOT_META}/meta_clips_info_fmin1_aes.csv --aesmin 5

# # 3.3 Predict optical flow scores.
torchrun --standalone --nproc_per_node 4 tools/scoring/optical_flow/inference.py /data/shanglinyuan/ysl-opensora-data/datasets_csv/Sports-1M/subset17_cut/video_info_fmin1_aes.csv


# # 4.1 Generate caption. This should output ${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5_caption_part*.csv
torchrun --nproc_per_node 4 --standalone -m tools.caption.caption_llava /data/shanglinyuan/ysl-opensora-data/datasets_csv/Sports-1M/subset17_cut/video_info_fmin1_aes_flow_aesmin3.0_flowmin1.5.csv --dp-size 4 --tp-size 1 --model-path /data/shanglinyuan/ysl-opensora-data/tools/caption/pllava_dir/PLLaVA/MODELS/pllava-7b --prompt video

PYTHONPATH=$PYTHONPATH:/data/shanglinyuan/ysl-opensora-data/tools/caption/pllava_dir/PLLaVA python tools/caption/pllava_dir/caption_pllava.py --pretrained_model_name_or_path /data/shanglinyuan/ysl-opensora-data/tools/caption/pllava_dir/PLLaVA/MODELS/pllava-7b --use_lora --lora_alpha 4 --num_frames 4 --weight_dir /data/shanglinyuan/ysl-opensora-data/tools/caption/pllava_dir/PLLaVA/MODELS/pllava-7b --csv_path /data/shanglinyuan/ysl-opensora-data/datasets_csv/Sports-1M/subset17_cut/video_info_fmin1_fmax500_aes_flow_aesmin3.5_flowmin1.5_flowmax100.0.csv --pooling_shape 4-12-12 --batch_size 6

# # 4.2 Merge caption results. This should output ${ROOT_META}/meta_clips_caption.csv
# python -m tools.datasets.datautil ${ROOT_META}/meta_clips_info_fmin1_aes_aesmin5_caption_part*.csv --output ${ROOT_META}/meta_clips_caption.csv

# # 4.3 Clean caption. This should output ${ROOT_META}/meta_clips_caption_cleaned.csv
# python -m tools.datasets.datautil \
#   ${ROOT_META}/meta_clips_caption.csv \
#   --clean-caption \
#   --refine-llm-caption \
#   --remove-empty-caption \
#   --output ${ROOT_META}/meta_clips_caption_cleaned.csv

# # 4.4 Optionally generate tags (e.g., objects) based on the captions. This should output your_output_prefix_{key}.csv
# torchrun --nproc_per_node 8 --standalone -m tools.caption.caption_llama3 ${ROOT_META}/meta_clips_caption_cleaned.csv --key objects --output_prefix your_output_prefix
