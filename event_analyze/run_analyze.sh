# python event_analyze/analyze_episode.py \
#     /efs/share/1919650160032350208/efs_backup/nas-backup/compressed_data/astribot/dishwasher_2_fx_20260529_compressed/dishwasher_2_fx_20260529_episode_27.hdf5 \
#     --output event_analyze/output/dishwasher_2_fx_20260529_episode_27_analysis \
#     --top-k 24 \
#     --lerobot-episode 11 \
#     --lerobot-max-raw-frame 1645

export AIGC_API_KEY="msk-57d214329a4b63bebb0617ce7dadbddb4d00c382f61be32a015d73c4e6c8d121"
export AIGC_USER="suty11"
export AIGC_BASE_URL="https://aimpapi.midea.com/t-aigc/aimp-qwen3-5-122b/v1"

python event_analyze/qwen_verify_events.py \
    event_analyze/output/dishwasher_2_fx_20260529_episode_27_analysis \
    --task "put the dish into the dishwasher"