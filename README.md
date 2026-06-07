# 可信多模态交通 AI 系统

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyTorch](https://img.shields.io/badge/pytorch-2.6-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76b900)](https://developer.nvidia.com/cuda-toolkit)
[![GPU](https://img.shields.io/badge/GPU-RTX%204060-83b81a)](https://www.nvidia.com/)

**三阶段解耦架构的可信交通场景理解系统 —— 带 GPU 加速、碰撞检测、评价框架与 Web 可视化平台。**

感知 → 拓扑 → 叙事 → 评价，各阶段通过 pickle/JSON 松耦合，可独立验证。

> 传感与测试技术课程设计

---

## 系统架构

```
原始视频
   │
   ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 1 ── 双通道感知                                    │
│   Channel A: YOLO-BDD100K 检测 + BoT-SORT 跟踪           │
│   Channel B: Depth Anything v2 ViT-S 单目深度估计         │
│   ⚡ GPU 加速: ~7.5 fps (RTX 4060)                       │
└────────────────────────┬─────────────────────────────────┘
                         │ pickle
                         ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 2 ── 四阶段拓扑引擎 + 碰撞检测                      │
│   Phase 1: 双池分流（stable / suspended）                 │
│   Phase 2: LCSS 级联缝合 + Hungarian 全局最优分配         │
│   Phase 3: 轨迹精炼（线性插值 + 平滑 + 静止段检测）        │
│   Phase 4: 曲率积分 M2（自适应 ε 阈值判定转弯）            │
│   🆕 TTC 碰撞风险检测（2D 空间距离 + 严重程度分级）        │
└────────────────────────┬─────────────────────────────────┘
                         │ ST-Graph JSON
                         ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 3 ── 可信叙事                                     │
│   事实序列化 → LLM（qwen2.5:7b）→ 护栏防御 → 报告        │
│   🆕 碰撞风险事实自动注入叙事代理                          │
└────────────────────────┬─────────────────────────────────┘
                         │ 交通报告
                         ▼
┌──────────────────────────────────────────────────────────┐
│ 🆕 Stage 4 ── 系统评价                                  │
│   10 项指标 × 3 类别（数据质量/报告质量/端到端）           │
│   可视化图表（雷达图/热力图/柱状图/箱线图等 11 张）        │
│   HTML 报告 + 交互仪表板                                  │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ 🆕 输出                                                   │
│   标注视频 (H.264) · 交通报告 · 评价报告 · 终端日志       │
│   Web 可视化平台 (http://localhost:5000)                  │
└──────────────────────────────────────────────────────────┘
```

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| **GPU 加速** | RTX 4060 推理，~7.5 fps（CPU 的 8 倍），10 秒视频仅需 ~13 秒 |
| **碰撞检测** | 基于 2D 空间距离的 TTC 碰撞风险检测，Critical/High/Moderate 三级严重程度 |
| **评价框架** | 10 项评价指标，自动雷达图/热力图/箱线图，HTML 专业报告 |
| **Web 平台** | 浏览器上传视频 → 自动处理 → 标注视频 + 报告 + 图表一站式展示 |
| **标注视频** | H.264 编码，浏览器直接播放，含检测框/ID/深度/碰撞警告 |
| **终端日志** | 自动保存每次运行的完整终端输出为 `output/*.txt` |
| **可信护栏** | LLM 叙事禁用关键词扫描 + 否定前缀感知 + 事实绑定 |

---

## 仓库结构

```
├── run_pipeline.py             # 一键运行（S1→S2→S3→评价 + 终端日志）
├── perception_pipeline.py      # Stage 1：双通道感知层
├── topology_engine.py           # Stage 2：四阶段拓扑引擎 + 碰撞检测
├── narrative_agent.py           # Stage 3：可信叙事 + 护栏 + 碰撞事实注入
├── project_config.py            # 路径与模型配置
├── activate_env.ps1             # 🆕 一键激活虚拟环境（PowerShell）
├── requirements.txt             # Python 依赖
├── setup.bat                    # conda 环境配置（可选）
├── README.md                    # 本文件
├── LICENSE                      # MIT
│
├── evaluation/                  # 🆕 评价框架模块
│   ├── __init__.py
│   ├── config.py                # 评价配置
│   ├── metrics.py               # 10 项评价指标
│   ├── data_loader.py           # 数据加载器
│   ├── runner.py                # 评价运行引擎
│   ├── visualization.py         # 可视化图表生成
│   └── report.py                # HTML/PDF 报告生成
│
├── web_platform/                # 🆕 Web 可视化平台
│   ├── app.py                   # Flask 后端（上传/运行/结果 API）
│   ├── video_annotator.py       # 标注视频生成器（H.264）
│   ├── start_platform.bat       # 一键启动脚本
│   ├── templates/
│   │   └── index.html           # 深色主题仪表板 SPA
│   └── static/
│       ├── css/style.css         # 现代暗色主题
│       └── js/app.js             # 前端逻辑
│
├── models/                      # 模型权重（需手动下载）
│   ├── depth_anything_v2_vits.pth  (~95 MB)
│   └── best.pt                     (~20 MB)
│
├── Depth-Anything-V2/           # 深度估计库（git clone）
│
├── videos/                      # 样例视频
│   ├── S06_c043.avi
│   └── S06_c046.avi
│
├── output/                      # 🆕 运行输出
│   ├── stage1_result/           # Stage 1 pickle
│   ├── st_graph_*.json          # Stage 2 ST-Graph
│   ├── traffic_report_*.txt     # Stage 3 交通报告
│   ├── *_annotated.mp4          # 标注视频（H.264）
│   ├── pipeline_log_*.txt       # 终端日志
│   └── eval_output/             # 评价输出
│       └── <scene>/
│           ├── evaluation_results.json
│           ├── evaluation_report.html
│           ├── dashboard.html
│           └── figures/         # 11 张可视化图表
│
└── 最终材料/                    # 课程交付物
    ├── 01–08 说明文档
    ├── charts/
    └── traffic_report_*.txt
```

---

## 关键设计决策

| 决策 | 理由 |
|------|------|
| **三阶段解耦** | 各阶段独立可验证；pickle / JSON 作为阶段间交换格式 |
| **双池分流**（Phase 1） | 稳定轨迹（≥15 帧，平均置信度 ≥0.4）与噪声片段分离 |
| **LCSS 级联缝合**（Phase 2） | 对短暂遮挡鲁棒；Hungarian 算法求全局最优分配 |
| **曲率积分 M2**（Phase 4） | `W = Σ ω_t`，通过 `np.gradient` 计算；ε = P85(\|W\|)，自适应 |
| **2D 空间 TTC**（碰撞检测） | 基于图像坐标 (cx,cy) 二维欧氏距离，替代纯深度比较；150px 空间邻近阈值过滤假阳性 |
| **碰撞严重程度分级** | Critical (<0.3s TTC) / High (<0.8s) / Moderate，按车辆对去重 + 1 秒窗口合并 |
| **LLM 护栏**（Stage 3） | 禁用关键词扫描 + 否定前缀感知 + 事实绑定 |
| **零微调** | 全部模型使用预训练权重 —— YOLO-BDD100K、Depth Anything v2、Qwen2.5 |

---

## 🚀 快速开始

### 环境要求

| 组件 | 配置 |
|------|------|
| Python | 3.10+ |
| PyTorch | 2.6+ CUDA 12.4（GPU）或 CPU |
| GPU（推荐） | NVIDIA RTX 4060 8GB+（CPU 可回退，~8× 慢） |
| LLM（Stage 3） | Ollama + Qwen2.5:7b |
| 操作系统 | Windows 10/11 |

### 1. 激活虚拟环境

项目自带 `.venv` 虚拟环境，已包含全部依赖。在项目根目录打开 PowerShell：

```powershell
.\activate_env.ps1
```

或手动激活：

```powershell
.venv\Scripts\Activate.ps1
$env:YOLO_CONFIG_DIR = (Get-Location).Path
```

### 2. 克隆 Depth Anything v2

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
```

> 已克隆则跳过。

### 3. 下载模型权重

确保 `models/` 目录存在以下文件：

| 文件 | 大小 | 用途 |
|------|------|------|
| `depth_anything_v2_vits.pth` | ~95 MB | 深度估计 |
| `best.pt` | ~20 MB | YOLO 车辆检测 |

### 4. 安装 Ollama（Stage 3 需要）

```bash
# 下载安装 https://ollama.com
ollama pull qwen2.5:7b
```

> Stage 1 + 2 可跳过。未安装时 Stage 3 自动降级为事实数据摘要。

### 5. 验证环境

```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# 应输出: CUDA: True
```

---

## 📖 使用指南

### 命令行运行

```bash
# 一键运行（自动发现 videos/ 下所有视频，含评价）
python run_pipeline.py

# 处理单个视频（10 秒快速测试）
python run_pipeline.py --video videos\S06_c046.avi --max-secs 10

# 跳过 Stage 1（使用已有 pickle）
python run_pipeline.py --skip-s1

# 跳过评价
python run_pipeline.py --no-eval
```

### Web 平台运行

```bash
# 启动平台
.venv\Scripts\python.exe web_platform\app.py
# 或双击
web_platform\start_platform.bat
```

浏览器打开 **http://localhost:5000**，上传视频 → 自动处理 → 查看结果：
- 🎬 **标注视频**：H.264 浏览器直接播放，含检测框 + 碰撞警告
- 📝 **交通报告**：LLM 生成的可信叙事报告
- 📈 **评价报告**：10 项指标 + HTML 专业报告
- 📊 **可视化图表**：11 张评价图表

### 分阶段运行

```bash
# 仅 Stage 1
python perception_pipeline.py

# 仅 Stage 2
python topology_engine.py output/stage1_result/frame_dicts_xxx.pkl

# 仅 Stage 3
python narrative_agent.py output/st_graph_frame_dicts_xxx.json
```

---

## 📊 性能对比

| 场景 | CPU (i7) | GPU (RTX 4060) | 加速比 |
|------|----------|----------------|--------|
| Stage 1 推理 | ~0.9 fps | **~7.5 fps** | **8×** |
| 10s 视频全流程 | ~120s | **~53s** | **2.3×** |
| 模型加载 | 8s | 3s | 2.7× |

---

## 🔍 碰撞检测

### 改进效果

| 指标 | 改进前 | 改进后 |
|------|--------|--------|
| 10s 视频碰撞事件 | 727 次（假阳性） | **14 次**（有意义） |
| TTC 计算 | 纯深度差 | **2D 欧氏距离 + 接近速度** |
| 空间过滤 | 无 | 150px 邻近阈值 |
| 严重程度 | 无 | **Critical / High / Moderate** 三级 |
| 事件去重 | 无 | 按车辆对去重 + 1s 窗口合并 |

### 输出示例

```
[严重] car_4 + car_10 | TTC: 0.08s ← 极危险
[严重] car_8 + car_10 | TTC: 0.10s
[严重] car_3 + car_10 | TTC: 0.17s
[高危] car_22 + car_62 | TTC: 0.64s
[中等] 4 对不同车辆对，共 6 次
```

---

## 📈 评价体系

10 项指标 × 3 个类别：

| 类别 | 指标 | 说明 |
|------|------|------|
| **数据质量** | Data_Integrity, Data_Consistency, Graph_Complexity, Node_Attribute_Richness | 评估 Stage 1 结构化数据质量 |
| **报告质量** | Report_Completion, Information_Density, Report_Readability, Event_Detection_Richness | 评估 Stage 3 叙事报告质量 |
| **端到端** | Data_Report_Alignment, System_Comprehensive_Score | 评估数据到报告的一致性 |

可视化输出：雷达图、热力图、柱状图、趋势分布图、相关性散点图、箱线图、综合对比图 (11 张) + HTML 仪表板 + HTML 评价报告。

---

## 📦 数据集

本项目使用 [NVIDIA AI City Challenge 2022](https://www.aicitychallenge.org/2022-challenge/) Track 1 数据集。

| 视频 | 大小 | 时长 | 来源 |
|------|------|------|------|
| `S06_c043.avi` | 100 MB | ~2 min | 仓库内置 |
| `S06_c046.avi` | 63 MB | ~1 min | 仓库内置 |
| `S05_c028.avi` | 435 MB | ~7 min | AICity22 官网 |
| `S06_c041.avi` | 172 MB | ~3 min | AICity22 官网 |

> **引用：** M. Naphade et al., "The 6th AI City Challenge," CVPR Workshop, 2022.

---

## 常见问题

| 问题 | 解决方法 |
|------|----------|
| `torch.cuda.is_available() = False` | 重新安装 CUDA 版 PyTorch：`pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu124` |
| `No module named 'cv2'` | 运行 `.\activate_env.ps1` 激活虚拟环境 |
| `YOLO model not found` | 确保 `models/best.pt` 存在 |
| `Depth weights not found` | 确保 `models/depth_anything_v2_vits.pth` 存在 |
| `No module named 'depth_anything_v2'` | 执行 `git clone https://github.com/DepthAnything/Depth-Anything-V2` |
| 浏览器无法播放视频 | 需安装 FFmpeg（已内置 imageio-ffmpeg），或直接下载用 VLC 播放 |
| 网页打不开 (500) | 检查端口 5000 是否被占用：`netstat -ano \| findstr 5000` |
| 处理速度慢 | 确认 GPU 可用：`python -c "import torch; print(torch.cuda.is_available())"` |

---

## 许可证

MIT — 详见 [LICENSE](LICENSE)。
