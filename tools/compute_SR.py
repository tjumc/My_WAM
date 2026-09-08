import re
import json
from pathlib import Path

ROOT = Path("/home/ubuntu/zhangjj/FastWAM/evaluate_results/robotwin/robotwin_uncond_3cam_384")
OUT = "success_rates_summary.json"

pattern = re.compile(
    r"done task=(?P<task_name>\S+)\s+phase=(?P<phase>\S+)\s+gpu=\d+\s+success_rate=(?P<success_rate>[0-9.]+)"
)

# 按 task_name 去重，保留最高 success_rate
best_records = {}

for subdir in sorted(ROOT.iterdir()):
    if not subdir.is_dir():
        continue

    log_file = subdir / "manager.log"
    if not log_file.exists():
        continue

    text = log_file.read_text(encoding="utf-8", errors="ignore")

    for m in pattern.finditer(text):
        task_name = m.group("task_name")
        success_rate = float(m.group("success_rate"))

        if task_name not in best_records:
            best_records[task_name] = {
                "task_name": task_name,
                "success_rate": success_rate
            }
        else:
            # 只保留最高分
            if success_rate > best_records[task_name]["success_rate"]:
                best_records[task_name]["success_rate"] = success_rate

# 转成列表，按任务名排序，输出更稳定
records = sorted(best_records.values(), key=lambda x: x["task_name"])

total = sum(item["success_rate"] for item in records)
num = len(records)
avg = total / num if num else 0.0

output = {
    "records": records,
    "summary": {
        "num": num,
        "total": total,
        "avg": avg
    }
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Saved to: {OUT}")
print(json.dumps(output["summary"], ensure_ascii=False, indent=2))