# V2

历史实验版本：conservative atomic-event verification。

该版本产生了过度收缩（当前 dishwasher 样例中 24 个 candidate 最终为 0 个 atomic event），因此保留用于 error analysis，不再作为当前运行入口。

代码：
- `qwen_verify_events_v2.py`
- `qwen_compose_phases_v2.py`
- `refine_boundaries_v2.py`

历史结果位于：

`output/<episode>_analysis/versions/v2/`
