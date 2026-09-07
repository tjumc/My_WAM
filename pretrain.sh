cd /data/share/1919650160032350208/wzy/FastWAM
torchrun --standalone --nproc_per_node=16 scripts/precompute_text_embeds.py task=pick_place_1e-4
python scripts/precompute_stats_optimize.py task=pick_place_1e-4 --output runs/pick_place_stats.json