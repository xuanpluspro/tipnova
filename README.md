# CGTipNet

CGTipNet 是基于 SAHCC 的植物叶尖定位与计数模型。它默认启用跨尺度上下文引导的细节融合：在 stride-8 的 C4 上提取局部细节，由 stride-16 的 C5 生成门控，然后将增强后的 C4 与 C5 融合。保留 SAHCC 的锚点、点预测头、匹配器和损失，默认新增 24,608 个参数。训练速度和显存需实测。

默认变体为 `--variant cgtipnet`；`--variant tipfuse` 是旧名称的兼容别名。`--variant sah` 是同仓库 SAHCC 对照，`--variant ddg` 是原 TipNova 方向细节门控对照。独立的 leaf-tip-inference 仓库需同步结构后才可使用 CGTipNet 权重。

设计参考：[UniGC 的 CSD 实现](https://github.com/xiaofengxie-dl/UniGC/blob/main/models/csd_p2pnet/models/csd_block.py)。

数据集 RiceW3999：FoMo4Wheat Rice Leaf 1,700 张水稻图像、Li 等（2023）500 张小麦图像、Yang 等（2025）1,799 张小麦图像。沿用原划分（训练 3,429 张，验证 570 张），支持分来源指标。数据与权重不随仓库分发。

原 DDG 对照结构见 [架构图](docs/tipnova_architecture.svg)。CGTipNet 的精度、吞吐和显存尚未实测。

## GPUFree 安装

在已有 torch >= 2.3 的环境中运行，不重新安装镜像自带的 torch。无需 torchvision、Lightning、Hydra 或 CUDA 自定义扩展。

```bash
cd /root/gpufree-data
git clone https://github.com/xuanpluspro/tipnova.git
cd tipnova
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 已有 uv 时跳过第一行
python -m pip install uv -i https://mirror.sjtu.edu.cn/pypi/web/simple
uv pip install --python "$(command -v python)" -r requirements.txt \
  --index-url https://mirror.sjtu.edu.cn/pypi/web/simple
```

## 数据和预训练

已有数据可以直接复用，不必移动或复制。默认目录结构：

```text
data/RiceW3999/
  JPEGImages/<完整图片名>.jpg 或 .png
  PointAnnotations/<同名>.txt
  ImageSets/train.txt
  ImageSets/val.txt
pretrained/vgg16_bn-6c64b313.pth
```

每个标注文件每行 `x y`，单位为原图像素；空文件表示没有叶尖。清单每行是完整文件名主干，不带扩展名；中文和空格完整保留。不能使用按空白切分清单的方式。坐标必须有限且满足 `0 <= x < W, 0 <= y < H`，重复坐标会明确报错。支持 jpg/jpeg/png/bmp/tif/tiff，扩展名大小写兼容。

使用与 PET/P2RLoss 相同的 ImageNet VGG16-BN 权重：

```bash
cp /root/gpufree-share/vgg16_bn-6c64b313.pth pretrained/
python check_data.py --data-path /root/gpufree-data/p2rloss/data/RiceW3999
```

快速检查包括图片头可读性、配对、点边界、划分 ID 重叠；`--check-overlap` 还检查解码后完全重复图片。不同裁块来自同一原图的潜在泄漏仍需按原图来源审计，不能依靠像素哈希排除。

## 开始训练

```bash
DATA_PATH=/root/gpufree-data/p2rloss/data/RiceW3999 \
BATCH_SIZE=32 EPOCHS=50 WORKERS=4 \
bash train_ricew3999.sh ricew3999_cgtipnet_ep50_bs32_run1
```

使用本仓库 data 下的数据时，可以省略 DATA_PATH。推荐在已安装的 tmux 会话中启动。默认 batch 32 是为你的 48GB 显存设置的起点，并非已测显存保证；OOM 时改为 16，并使用新实验名。可追加 `--amp` 开启 CUDA FP16；默认 FP32。不要运行中途修改训练代码或升级 torch。

| 参数 | 默认值 |
|---|---|
| 训练／验证输入 | 每图在线随机一个 512×512 裁块／原尺寸整图 |
| batch／总 epochs／workers | 32／50／4 |
| 验证 | 每个完整 epoch，batch=1（VAL_EVERY 默认 1） |
| 优化器 | AdamW，weight decay=1e-4 |
| 主干／其余学习率 | 1e-5／1e-4 |
| 梯度裁剪 | 全局范数 0.1 |
| StepLR | step_size=3500，gamma=0.1；50 轮内不会衰减 |
| 参考点 | stride 8，每格 2×2 个点，偏移倍率 100 |
| 匹配 | SAH，使用 anchor_points；分类／几何代价系数 1／1 |
| 局部间距 | 最近 4 个其他标注点平均距离，kappa=1.2，alpha_min=0.03 |
| 损失 | 前景／背景加权 sigmoid BCE：1／0.5；位置平方误差系数 0.0002 |
| 前景阈值／随机种子 | 0.5／2024 |

局部间距在原始整图点集合上计算，缩放后按完整缩放点集合计算精确距离，再与标注一起筛选裁块；不在裁块内重新寻找邻居，避免边缘处邻居被裁掉而夸大间距。空标注正常训练，单点图使用官方 1 像素间距约定并随缩放调整。不会生成高斯密度图。

随机缩放范围 0.7–1.3，保证短边不小于裁块尺寸；随机水平翻转。512 原图为保证可裁块，实际缩放会限制在不缩小的范围，1024 图则可在完整范围缩放。每轮是每张原图抽取一次裁块，并非遍历该图所有区域。验证不缩放，只补齐到 16 的倍数，并移除原图边界外的预测。没有距离 NMS，以免合并相邻的真实叶尖。

### 多目录混合训练

每个根目录保留自己的 `JPEGImages/`、`PointAnnotations/`、`ImageSets/train.txt` 和 `ImageSets/val.txt`。程序拼接样本清单，使用一个打乱的训练集和一个验证集；无需复制图片、合并目录或改名。验证 MAE、RMSE、F1 按合并后的全部样本统一计算，仍会单独输出来源诊断表。默认按样本数量采样。

```bash
DATA=/work/home/mx123/RiceAndWheatSet
python check_data.py --data-path "$DATA/ricetip" "$DATA/wheat500"
BATCH_SIZE=32 EPOCHS=50 bash train_ricew3999.sh rice_wheat_cgtipnet \
  --data-path "$DATA/ricetip" "$DATA/wheat500"
```

将路径换成服务器上的实际目录名。单目录仍可用原命令。多个目录也可传给 `evaluate.py --data-path`。续训时保持目录和参数顺序一致。不要同时传入 RiceW3999 和其已包含的子集；如怀疑跨来源重复，可先对多个目录运行 `check_data.py --check-overlap`。该检查只能发现完全相同的解码像素，重叠裁块仍需检查原图来源。

原图尺寸可以不同。训练在线缩放和随机裁成统一的 512×512（可通过 `--patch-size` 改为不小于 32 的 16 的倍数），同步变换点坐标；验证每次处理一张原尺寸图，右下补齐到 16 的倍数且每边至少 32 像素，计算指标时使用原图范围。

## 断点恢复

同一条命令、同一实验名再次执行，会自动恢复 `output/checkpoint.pth`，不要求再次读取 ImageNet 权重。也可以显式指定：

```bash
python train.py --data-path /path/RiceW3999 --tag original_tag \
  --resume exp/original_tag/output/checkpoint.pth
```

显式恢复时其余训练参数需与原实验相同。保存模型、AdamW、StepLR、AMP scaler、随机状态、最佳结果和数据指纹。每个完整 epoch 后原子保存；断电／OOM 导致当前 epoch 中断时，从上个完整 epoch 重跑。不是 batch 级恢复。总 epochs、匹配器等关键配置变更会拒绝恢复；新实验使用新 tag。batch/workers 改变允许恢复但会警告轨迹变化。

## 输出文件

`exp/<tag>/output/` 包含：

- `checkpoint.pth`：最新完整训练状态，每轮更新。
- `best_checkpoint.pth`：验证 MAE 最低的完整训练状态，包含一个预测模型；没有教师模型。
- `config.json`、`resume_config.json`、`train.log`：配置、恢复配置、日志。
- `train_history.csv`：损失、学习率、训练耗时和 GPU 分配显存峰值。
- `metrics_history.csv`、`best_metrics.json`：验证历史和最佳验证指标。
- `evaluation/epoch_XXXX/metrics.json`、`per_image.csv`、`per_source.csv`：协议、逐图结果及来源分组统计。

启动脚本另追加保存 `exp/<tag>/train.log/running.log`。MAE/RMSE 越低越好；Precision、Recall、F1@4/8/16px 越高越好，比例范围 0–1。定位评估最大化容差内一对一匹配数，再最小化距离，汇总全数据集 TP/FP/FN。单目录时按文件名前缀 `__` 之前提取来源，多目录时按数据根目录分组，仅用于分析，不改变训练划分。JSON 指标位于 `metrics` 字段。

## 单独评估和推理

```bash
python evaluate.py --checkpoint exp/<tag>/output/best_checkpoint.pth \
  --data-path /root/gpufree-data/p2rloss/data/RiceW3999 --output evaluation/best

python infer.py --checkpoint exp/<tag>/output/best_checkpoint.pth \
  --input /path/ryegrass --output predictions/ryegrass

# 大图显存不足时：重叠区按中线划归所属块，不做半径去重
python infer.py --checkpoint exp/<tag>/output/best_checkpoint.pth \
  --input /path/ryegrass --output predictions/ryegrass_tiles --tile-size 1024 --overlap 64
```

推理输出每图 xy TXT、含置信度 JSON、标点图片以及 counts.csv。整图与滑窗不保证预测一致，比较模型时固定推理协议。当前 loader 只加载本工程导出的 checkpoint；官方 Lightning `.ckpt` 不能直接使用。

## 匹配器消融

网络结构完全相同，独立新 tag，保持种子／数据／训练轮数等一致：

```bash
# SAH + 锚点：默认
bash train_ricew3999.sh sah_anchor --variant sah
# 普通匹配 + 锚点：隔离匹配代价影响
bash train_ricew3999.sh baseline_anchor --matcher baseline --point-source anchor_points --cost-point 0.05
# 普通匹配 + 预测点：官方普通匹配默认
bash train_ricew3999.sh baseline_pred --matcher baseline --point-source pred_points --cost-point 0.05
# SAH + 预测点：检验匹配参考的影响
bash train_ricew3999.sh sah_pred --point-source pred_points
```

上述命令默认从本仓库 data 读取；外部数据时同样前置 DATA_PATH。普通匹配默认几何系数 0.05，SAH 默认 1，这是官方两种代价的设置，不能把差异全归于一个权重项；应进一步对固定权重 0.01/0.04/0.05/0.07 做扫描。`--neighbors`、`--kappa`、`--alpha-min` 可独立消融。论文的模块插入实验没有验证 APGCC+SAH，本仓库也不声称已完成该组合。

## 参数量、耗时及测试

```bash
python benchmark_model.py --checkpoint exp/<tag>/output/best_checkpoint.pth --size 512
# 可选 FLOPs 分析（fvcore 单独安装，不影响训练）
uv pip install --python "$(command -v python)" fvcore --index-url https://mirror.sjtu.edu.cn/pypi/web/simple
python benchmark_model.py --checkpoint exp/<tag>/output/best_checkpoint.pth --size 512 --flops
python -m unittest discover -s tests -v
```

基准默认 batch=1、FP32、20 次预热／100 次计时，排除读图、传输、匹配和可视化；报告参数量、latency ms、FPS、CUDA 显存。FLOPs 存在不支持的算子时只报告 counted_gflops，不将其冒充完整 FLOPs。SAH 不参与推理，因此同一网络替换匹配器不会增加推理参数。

出处、改动边界见 [REPRODUCTION.md](REPRODUCTION.md)，许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。


验证间隔可通过 `VAL_EVERY`／`--eval-every` 覆盖，旧检查点恢复时允许切换并记录日志，不会补评估历史轮次。切换频率可能改变随机数消耗和后续训练轨迹。


## 快速数据检查（RiceW3999 / VOC 点标注）

```bash
python check_data.py --data-path data/RiceW3999                 # 默认快速检查
python check_data.py --data-path data/RiceW3999 --decode        # 完整解码，检查像素损坏
python check_data.py --data-path data/RiceW3999 --check-overlap # 完整解码并按像素查重
```

快速模式只读取图片头中的尺寸及全部标注，检查配对、完整中文/空格 ID、重复 ID、坐标有限性/边界和 train/val/test ID 重叠。
不读取全部像素，因此不能保证发现图像数据区损坏；完整检查需加 `--decode`。
`--check-overlap` 比较解码后 RGB 像素，不能排除来自同一原图的不同重叠裁块泄漏。
JSON 仍包含图片数、点数、空标注数和尺寸统计，进度写入 stderr，`--no-progress` 可关闭。
检查仅依赖 Pillow 和 Python 标准库，不导入训练模型/PyTorch；有 tqdm 时显示进度条，没有时输出计数进度。
脚本只读，不去重、不删改数据，也不会自动加入每轮训练。只适用于 JPEGImages/PointAnnotations/ImageSets 点标注目录，不替代原始 SHHA 等格式的预处理。
本次每仓库 9 项针对性数据检查测试通过；未重跑模型训练测试。

SAH 保留训练 float32 精度下的重复坐标检查，错误会显示具体文件及行号；检查阶段无需计算 kNN。训练时的局部间距计算未改变。


## TipNova 与原型对照

在 C4 的 1×1 投影后、与上采样 C5 相加前，对 C4 分别做 1×5 和 5×1 深度卷积。两路响应取绝对值后相加，再以 1×1 卷积生成单通道门控图；增益为 `2 * sigmoid(gate)`，逐位置调节 C4。最后一层零初始化，初始增益为 1。默认 256 通道新增 2,817 个参数。

动机：细长叶片及其尖端的局部方向细节可能在跨尺度相加时被弱化。只改变 C4 特征的空间权重，保持 SAH 匹配、锚点、点回归头和损失不变。这是待验证的设计，不保证精度提高；轴向卷积也不具有旋转不变性。

```bash
BATCH_SIZE=40 EPOCHS=50 bash train_ricew3999.sh sah_baseline_ep50_bs40 --no-directional-detail
BATCH_SIZE=40 EPOCHS=50 bash train_ricew3999.sh tipnova_ep50_bs40 --variant ddg
```

比较同一数据划分、训练配置和验证协议下的 MAE、RMSE、F1@4/8/16px 及推理耗时。指定 `--variant ddg` 开启 DDG；旧基线检查点仍可加载评估，续训需显式传入 `--no-directional-detail`。新变体从相同 ImageNet 初始化独立训练，续训须保留开关。仓库内 evaluate.py / infer.py 从检查点自动恢复结构。独立 leaf-tip-inference 仓库需单独同步结构后才能加载变体权重。
