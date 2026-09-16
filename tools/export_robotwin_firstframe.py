from pathlib import Path
import subprocess

from fastwam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDatasetMetadata

ROOT = Path("/efs/share/1919650160032350208/projects/foundation_model/wzy/FastWAM/data/robotwin2.0/robotwin2.0")
EPISODES = [550, 599, 600, 4950, 4999, 5000]
OUT_DIR = Path("robotwin_clean_random_check")
 
meta = LeRobotDatasetMetadata(repo_id=str(ROOT), root=ROOT)
camera_key = next(k for k in meta.camera_keys if k.endswith("cam_high"))

OUT_DIR.mkdir(exist_ok=True)

for ep in EPISODES:
    video_path = meta.get_video_file_path(ep, camera_key)
    full_path = ROOT / video_path
    out = OUT_DIR / f"episode_{ep}.jpg"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hwaccel", "none",
            "-i", str(full_path),
            "-frames:v", "1",
            str(out),
        ],
        check=True,
    )

    print(out)