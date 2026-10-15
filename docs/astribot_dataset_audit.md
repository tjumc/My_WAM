# Astribot 全任务数据盘点

在已有WAM环境运行 `bash scripts/audit_astribot_catalog.sh`。不加载权重、不用GPU、
不操作真机、不预计算训练归一化统计、不修改catalog、不划分train/val/test。

## 输入范围和参数

`CATALOG` 指向 `configs/data/astribot_dataset_catalog.sh`。工具使用Bash 4+读取其中
`ASTRIBOT_DATASET_TASKS` 和 `ASTRIBOT_DATASET_CATALOG`，不是另维护一份目录列表。
这会source指定脚本，因此只能使用你信任的仓库脚本。

- `TASKS=()` 默认全部任务。单任务试运行可设 `TASKS=(dishwasher)`。
- `DATA_ROOT` / `BENCHMARK_ROOT` 为空时沿用catalog；非空只重映射本进程中的相应根路径。
  不修改catalog、源数据或已有输出路径；两个根目录分别处理，不能混用。
- `NUMERIC_SCAN=true` 扫描每个目录全部Parquet数值列。逐文件分批、串行读，不解码完整视频。
- 每目录随机抽 `PREVIEW_EPISODES_PER_DIR=3` 条轨迹，按 `PREVIEW_FRAMES=5` 个均匀时刻采样。
  抽样依赖种子和数据目录ID；固定目录/种子下，不受选择其他任务影响。更换挂载根可能改变ID。
  概览包括开始/过程/结尾，但均匀采样不是阶段分割，三条也不保证代表整个目录。
- `MAX_CAMERAS=3` 只限制图像概览的行数，元信息报告所有发现的相机。
- `PREVIEW_EPISODES_PER_DIR=0` 关闭预览。只扫元信息时，同时设 `NUMERIC_SCAN=false`。
- `.002`秒时间容差不会自动放宽。v3按相机的`from_timestamp`加偏移后查询视频。

依赖：Bash>=4、Python>=3.10、numpy、pyarrow；预览另需Pillow和PyAV（av）。
元信息支持`meta/episodes.jsonl`和`meta/episodes/**/*.parquet`。与仓库一样优先JSONL；
两种同时存在会提示检查旧副本。任务文本优先tasks.parquet，再用tasks.jsonl。
缺少标准LeRobot结构的目录只报告错误，不按25D猜测；不同state/action维度分别报告。
未知字段用dim_000等命名，不自动给未确认的维度贴上关节/夹爪语义。

## 输出与存储

默认新增 `ambiguity_outputs/dataset_audit/本次时间戳/`：

|文件|内容|
|---|---|
|report.md|中文总览和覆盖限制|
|tasks.csv|按任务汇总目录、元信息轨迹量和数值有效轨迹量|
|directories.csv|每目录规模、时长分布、相机/维度、存储指令、预览episode编号|
|numeric_summary.csv|每目录逐数值维度的范围、均值/标准差、相邻变化|
|duplicate_candidates.csv|完整state/action/timestamp精确重复的轨迹候选，不比较RGB|
|issues.csv|缺失目录、格式/时间戳/数值异常、预览失败|
|review.csv|人工填写完整/局部任务、物体种类数量、场景变化；初始为unknown|
|manifest.json、COMPLETE.json|输入清单、路径覆盖、种子、源码/元信息来源与运行状态|
|_local/index.html|本地查看所有抽样概览图；不上传Git|
|_local/目录ID/|逐episode统计、逐维CSV、元信息和抽样JPG，仅本地保留|

运行时 `.gitignore` 白名单只允许精简报告进入Git，`_local`、run.log、launcher.sh默认忽略。
不要 `git add -f`；详细产物另行备份。工具不复制原始视频或完整state/action数组。
每次新建目录，不覆盖已存在的结果；已有mining/inference/replay路径不改。

**盘点输出不直接用作训练归一化统计**。之后冻结划分后，仅从训练部分重新计算。

## 解释边界

人员/日期/目录名仅是来源线索，不等于不同场景。短于5秒不代表坏数据；任务文本也不一定
精确描述轨迹内容，是否包含完整八阶段等需看概览及必要时的原视频。

`low_change_step_fraction` 是逐维相邻差值<=阈值的比例；`longest_low_state_change_seconds`
是全部state维度变化都较小的最长连续段。它们采用原始单位，只作数据复核线索。
**恒定非零速度也会低变化，不能称为静止/停止/成功。**不会自动标注夹爪开合或抓取成功。

数值重复哈希按episode内部帧顺序包含真实state、action、timestamp及形状，排除目录和episode编号。
仅为数值精确相同候选：RGB可以不同，近重复/不同时间偏移/重新编码的近似数值不会匹配。
不自动删除候选；发现同一resolved路径重复出现在catalog时只扫描一次并记录，不重复累加数量。
目录内数值批次按frame排序；若跨批次/文件仍出现逆序，报告问题并不认证哈希/完整时序，
不会把错误索引替换为随机样本或补零。

`COMPLETE.json` 只表示遍历结束。`complete_with_issues`（退出码2）表示有问题需要看issues.csv；
缺少COMPLETE表示程序中断或致命错误。数值扫描报错则统计为partial，不拿未读部分当正常。
图像只使用经Parquet验证的真实episode/frame/timestamp，不拿名义帧率代替缺失的定位信息。
当前只解码磁盘视频；嵌入Parquet的图片会标记预览不可用，不阻断其他目录统计。

## 本地测试

`python -m pytest tests/test_audit_astribot_catalog.py -q`

测试以小型合成记录和模拟Parquet/视频接口为主，不声称已扫描服务器真实数据。
