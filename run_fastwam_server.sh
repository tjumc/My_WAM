python experiments/robotwin/fastwam_server.py \
    --checkpoint "/media/jz08/49630fca-f8b9-4c76-a173-2bcf51fee8a9/wam_ckpt/robotwin_proxy_lambda_video1/checkpoints/weights/step_003375.pt" \
    --dataset_stats "/media/jz08/49630fca-f8b9-4c76-a173-2bcf51fee8a9/mc/My_WAM/runs/robotwin_proxy_norm_stats.json" \
    --host 127.0.0.1 \
    --port 8765 \
    --device cuda \
    --vae_device_mode gpu