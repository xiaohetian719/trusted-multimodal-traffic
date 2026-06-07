# 可信多模态交通 AI 系统

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-red)](https://pytorch.org/)

**三阶段解耦架构的可信交通场景理解系统。**
感知 → 拓扑 → 叙事，各阶段通过 pickle/JSON 松耦合，可独立验证。

> 传感与测试技术课程设计

---

## 系统架构

```
原始视频
   │
   ▼
Stage 1 ──── 双通道感知 ──── Channel A: YOLO-BDD100K 检测 + BoT-SORT 跟踪
    │                         Channel B: Depth Anything v2 ViT-S 单目深度估计
    │
   pickle
    │
   ▼
Stage 2 ──── 四阶段拓扑引擎 ── Phase 1: 双池分流（stable / suspended）
    │                         Phase 2: LCSS 级联缝合 + Hungarian 全局最优分配
    │                         Phase 3: 轨迹精炼（线性插值 + 平滑 + 静止段检测）
    │                         Phase 4: 曲率积分 M2（自适应 ε 阈值判定转弯）
    │
   ST-Graph JSON
    │
   ▼
Stage 3 ──── 可信叙事 ──────── 事实序列化 → LLM（qwen2.5:7b）→ 护栏防御 → 报告
    │                         （12 个禁用词 + 否定前缀感知 + 最多 2 次重试）
    │
   ▼
 交通场景分析报告
```

## 仓库结构

```
├── run_pipeline.py             # 一键运行脚本（S1 → S2 → S3）
├── perception_pipeline.py     # Stage 1：双通道感知层
├── topology_engine.py          # Stage 2：四阶段拓扑引擎
├── narrative_agent.py          # Stage 3：可信叙事生成 + 护栏
├── project_config.py           # 路径与模型配置
├── requirements.txt            # Python 依赖
├── setup.bat                   # 一键配置 conda 环境
├── README.md                   # 本文件
├── LICENSE                     # MIT
├── .gitignore
│
├── videos/                     # 样例视频（完整原视频，~163 MB）
│   ├── S06_c043.avi
│   └── S06_c046.avi
│
└── 最终材料/                   # 课程交付物
    ├── 01–08 说明文档          # 设计方案与实现说明
    ├── S1/S2/S3 素材说明        # 各阶段素材索引
    ├── charts/                 # 可视化图表
    ├── traffic_report_*.txt    # LLM 生成的交通报告（Stage 3 输出）
    ├── st_graph_outputs/       # ST-Graph JSON（Stage 2 输出 → Stage 3 输入）
    └── 四段视频完整数据表.txt    # 四段视频的完整数据
```

> 视频文件、模型权重、开发工具、历史备份均未纳入版本控制（见 `.gitignore`）。

## 关键设计决策

| 决策 | 理由 |
|------|------|
| **三阶段解耦** | 各阶段独立可验证；pickle / JSON 作为阶段间交换格式 |
| **双池分流**（Phase 1） | 稳定轨迹（≥15 帧，平均置信度 ≥0.4）与噪声片段分离 |
| **LCSS 级联缝合**（Phase 2） | 对短暂遮挡鲁棒；Hungarian 算法求全局最优分配 |
| **曲率积分 M2**（Phase 4） | `W = Σ ω_t`，通过 `np.gradient` 计算；ε = P85(\|W\|)，自适应，截断至 [0.12, 1.5] rad |
| **LLM 护栏**（Stage 3） | 12 个禁用关键词，否定前缀感知扫描，最多 2 次拒绝重试 |
| **零微调** | 全部模型使用预训练权重 —— YOLO-BDD100K、Depth Anything v2、Qwen2.5 |

## M2 转弯语义

```
图像坐标系（y↓）：
  W > +ε  →  右转（顺时针）
  W < -ε  →  左转（逆时针）
  |W| ≤ ε →  直行

其中 W = 累积有向曲率，ε = |W| 的 85 百分位数
```

---

## 快速开始（零基础）

以下步骤适用于全新 Windows 机器，无需预先安装 Python。

### 1. 安装 Miniconda

下载并安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)（Python 3.10+，64 位）。

> 已安装 Anaconda 或 Miniconda 则跳过此步。

### 2. 克隆仓库

```bash
git clone https://github.com/Eternity-burial/trusted-multimodal-traffic-system.git
cd trusted-multimodal-traffic-system
```

### 3. 一键配置环境

双击 `setup.bat`，或在终端中运行：

```bash
setup.bat
```

脚本自动完成：
- 定位本机 conda 安装路径
- 创建 `ai_lab` 虚拟环境（Python 3.10）
- 安装全部 Python 依赖
- 安装 PyTorch（CUDA 12.1）
- 检查模型文件、Depth-Anything-V2、Ollama 状态

> 若提示 "conda not found"，从开始菜单打开 "Anaconda Prompt"，在该终端中执行后续命令。

### 4. 克隆 Depth Anything v2

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
```

### 5. 下载模型权重

创建 `models/` 目录，放入以下两个文件：

| 文件 | 大小 | 用途 | 来源 |
|------|------|------|------|
| `depth_anything_v2_vits.pth` | ~95 MB | Channel B 深度估计 | [Depth Anything v2 Releases](https://github.com/DepthAnything/Depth-Anything-V2/releases) |
| `best.pt` | ~20 MB | Channel A YOLO 检测 | [intelligent_traffic_ai](https://github.com/xiaohetian719/intelligent_traffic_ai)（`bdd_yolo_train_m_vs/weights/best.pt`） |

```bash
mkdir models
# 将上述两个文件放入 models/ 目录
```

### 6. 安装 Ollama（Stage 3 需要）

1. 下载并安装 [Ollama](https://ollama.com)
2. 拉取 Qwen2.5 模型：

```bash
ollama pull qwen2.5:7b
```

> 仅运行 Stage 1 和 Stage 2 可跳过此步。

### 7. 运行

```bash
# 激活环境
conda activate ai_lab

# 一键运行（自动发现 videos/ 下所有视频）
python run_pipeline.py

# 处理单个视频
python run_pipeline.py --video path/to/video.avi

# 快速测试：只处理前 30 秒
python run_pipeline.py --video path/to/video.avi --max-secs 30

# 跳过 Stage 1（使用已有 pickle）
python run_pipeline.py --skip-s1
```

**分阶段运行（调试用）：**

```bash
# 仅 Stage 1
python perception_pipeline.py

# 仅 Stage 2（需要 Stage 1 的 pickle）
python topology_engine.py output/stage1_result/frame_dicts_xxx.pkl

# 仅 Stage 3（需要 Stage 2 的 JSON）
python narrative_agent.py st_graph_output_xxx.json
```

---

## 运行环境

| 组件 | 配置 |
|------|------|
| Python | 3.10+（conda 环境 `ai_lab`） |
| PyTorch | 2.x（CUDA 12.1） |
| YOLO | BDD100K（10 类，仅保留车辆） |
| 深度估计 | Depth Anything v2 ViT-S（~95 MB） |
| LLM | Ollama + Qwen2.5:7b（Q4_K_M） |
| GPU | RTX 4060 8 GB（CPU 可回退运行） |

---

## 数据集

本项目使用 [NVIDIA AI City Challenge 2022](https://www.aicitychallenge.org/2022-challenge/)（AICity22）Track 1 数据集。

仓库包含其中 2 段完整视频作为直接可用的样例：

| 视频 | 大小 | 时长 | 内容 | 来源 |
|------|------|------|------|------|
| `S06_c043.avi` | 100 MB | ~2 min | 交叉路口车流 | 仓库内置 |
| `S06_c046.avi` | 63 MB | ~1 min | 交叉路口短时场景 | 仓库内置 |

另外 2 段视频因体积超过 GitHub 100 MB 限制，需从 AICity22 官网下载后放入 `videos/1/`：

| 视频 | 大小 | 时长 | 下载 |
|------|------|------|------|
| `S05_c028.avi` | 435 MB | ~7 min | [AICity22 官网](https://www.aicitychallenge.org/2022-data-and-evaluation/) |
| `S06_c041.avi` | 172 MB | ~3 min | [AICity22 官网](https://www.aicitychallenge.org/2022-data-and-evaluation/) |

> 完整数据集需从 AICity22 官方渠道申请下载，详见 [aicitychallenge.org](https://www.aicitychallenge.org/)。

**引用：**
> M. Naphade et al., "The 6th AI City Challenge," CVPR Workshop, 2022.

---

## 常见问题

| 问题 | 解决方法 |
|------|----------|
| `conda not found` | 从开始菜单打开 "Anaconda Prompt"，在该终端中执行后续命令 |
| `No module named 'cv2'` | 运行 `setup.bat`，或 `conda activate ai_lab && pip install -r requirements.txt` |
| `YOLO model not found` | 确保 `models/best.pt` 存在，或修改 `project_config.py` 中的路径 |
| `Depth weights not found` | 确保 `models/depth_anything_v2_vits.pth` 存在 |
| `No module named 'depth_anything_v2'` | 确保已在项目根目录执行 `git clone https://github.com/DepthAnything/Depth-Anything-V2` |
| `Ollama 服务不可用` | 启动 Ollama 桌面应用，确认 `ollama list` 中可见 `qwen2.5:7b` |
| 无 CUDA | CPU 可运行，但较慢（~0.3 fps vs GPU ~30 fps） |

## 许可证

MIT — 详见 [LICENSE](LICENSE)。
