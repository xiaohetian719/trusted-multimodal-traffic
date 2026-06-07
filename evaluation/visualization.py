"""
可视化模块
生成各种图表和仪表板，包括雷达图、热力图、柱状图等
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle
import seaborn as sns
from matplotlib import rcParams

# 配置中文字体
import matplotlib.font_manager as fm

# 全局中文字体属性
chinese_font_prop = None

def setup_chinese_font():
    """设置中文字体"""
    global chinese_font_prop
    
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/msyh.ttf',
        'C:/Windows/Fonts/simsun.ttc',
        '/Library/Fonts/SimHei.ttf',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                chinese_font_prop = fm.FontProperties(fname=font_path)
                font_name = chinese_font_prop.get_name()
                
                # 全面配置matplotlib字体
                rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial Unicode MS']
                rcParams['font.family'] = 'sans-serif'
                rcParams['font.size'] = 12
                rcParams['axes.labelsize'] = 12
                rcParams['axes.titlesize'] = 14
                rcParams['axes.unicode_minus'] = False
                rcParams['legend.fontsize'] = 10
                rcParams['xtick.labelsize'] = 10
                rcParams['ytick.labelsize'] = 10
                rcParams['figure.titlesize'] = 16
                
                print(f"[INFO] 成功加载中文字体: {font_name}")
                return
            except Exception as e:
                print(f"[WARNING] 加载字体失败 {font_path}: {e}")
    
    # 如果找不到中文字体，使用系统默认字体
    rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    rcParams['font.family'] = 'sans-serif'
    rcParams['axes.unicode_minus'] = False
    print("[WARNING] 未找到中文字体，使用默认字体")

setup_chinese_font()

def get_chinese_font():
    """获取中文字体属性"""
    return chinese_font_prop

from .runner import EvaluationRunner, SceneEvaluationResult
from .config import VisualizationConfig

logger = logging.getLogger(__name__)


class VisualizationConfig:
    """可视化配置类（向后兼容）"""

    def __init__(self,
                 figure_dpi: int = 300,
                 figure_size: Tuple[int, int] = (12, 8),
                 color_palette: str = 'husl',
                 style: str = 'seaborn-v0_8-darkgrid',
                 save_format: str = 'png'):
        """
        初始化可视化配置

        Args:
            figure_dpi: 图片DPI分辨率
            figure_size: 图片默认大小
            color_palette: 调色板
            style: matplotlib样式
            save_format: 保存格式 ('png', 'pdf', 'svg')
        """
        self.figure_dpi = figure_dpi
        self.figure_size = figure_size
        self.color_palette = color_palette
        self.style = style
        self.save_format = save_format

        # 设置样式
        try:
            plt.style.use(style)
        except:
            logger.warning(f"样式 {style} 不可用，使用默认样式")

        sns.set_palette(color_palette)


class RadarChartVisualizer:
    """雷达图可视化"""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def generate_scene_radar(self,
                             scene_name: str,
                             result: SceneEvaluationResult,
                             output_path: Path) -> Path:
        """
        生成单个场景的雷达图

        Args:
            scene_name: 场景名称
            result: 场景评价结果
            output_path: 输出路径

        Returns:
            Path: 保存的图片路径
        """
        logger.info(f"生成场景 {scene_name} 的雷达图...")

        # 提取指标数据
        metrics_dict = result.get_metrics_dict()

        if not metrics_dict:
            logger.warning(f"场景 {scene_name} 无指标数据，跳过雷达图生成")
            return None

        # 按类别分组指标
        categories = {
            'data_quality': [],
            'report_quality': [],
            'end_to_end': [],
            'comprehensive': []
        }

        category_labels = {
            'data_quality': '数据质量',
            'report_quality': '报告质量',
            'end_to_end': '端到端性能',
            'comprehensive': '综合评分'
        }

        for category, metrics in result.metrics_results.items():
            if category in categories:
                for metric_name, metric_result in metrics.items():
                    categories[category].append(metric_result.metric_value)

        # 计算各类别平均值
        category_values = []
        category_names_display = []

        for category, values in categories.items():
            if values:
                avg_value = np.mean(values)
                category_values.append(avg_value)
                category_names_display.append(category_labels[category])

        if not category_values:
            logger.warning(f"场景 {scene_name} 无有效指标")
            return None

        # 创建雷达图
        fig, ax = plt.subplots(figsize=self.config.figure_size,
                               subplot_kw=dict(projection='polar'))

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(category_values), endpoint=False).tolist()
        category_values_plot = category_values + [category_values[0]]  # 闭合
        angles_plot = angles + [angles[0]]

        # 绘制雷达图
        ax.plot(angles_plot, category_values_plot, 'o-', linewidth=2, label=scene_name)
        ax.fill(angles_plot, category_values_plot, alpha=0.25)

        # 设置标签
        ax.set_xticks(angles)
        ax.set_xticklabels(category_names_display, size=10, fontproperties=chinese_font_prop)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], size=8)
        ax.grid(True)

        plt.title(f'{scene_name} - 评价指标雷达图', size=14, weight='bold', pad=20, fontproperties=chinese_font_prop)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        # 保存
        output_file = output_path / f's1_metrics_radar_{scene_name}.{self.config.save_format}'
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"雷达图已保存: {output_file}")
        return output_file

    def generate_multi_scene_radar(self,
                                   results: Dict[str, SceneEvaluationResult],
                                   output_path: Path) -> Path:
        """
        生成多场景对比雷达图

        Args:
            results: 所有场景的评价结果
            output_path: 输出路径

        Returns:
            Path: 保存的图片路径
        """
        logger.info("生成多场景对比雷达图...")

        # 收集所有场景的综合评分
        scene_names = []
        scores = []

        for scene_name, result in sorted(results.items()):
            scene_names.append(scene_name)
            scores.append(result.overall_score)

        if not scores:
            logger.warning("无有效的评分数据")
            return None

        # 创建雷达图
        fig, ax = plt.subplots(figsize=self.config.figure_size,
                               subplot_kw=dict(projection='polar'))

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(scores), endpoint=False).tolist()
        scores_plot = scores + [scores[0]]  # 闭合
        angles_plot = angles + [angles[0]]

        # 绘制雷达图
        ax.plot(angles_plot, scores_plot, 'o-', linewidth=2.5, markersize=8, label='综合评分')
        ax.fill(angles_plot, scores_plot, alpha=0.25)

        # 设置标签
        ax.set_xticks(angles)
        ax.set_xticklabels(scene_names, size=9)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], size=8)
        ax.grid(True)

        plt.title('多场景综合评分对比雷达图', size=14, weight='bold', pad=20, fontproperties=chinese_font_prop)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), prop=chinese_font_prop)

        # 保存
        output_file = output_path / f's1_metrics_radar_comparison.{self.config.save_format}'
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"多场景雷达图已保存: {output_file}")
        return output_file


class HeatmapVisualizer:
    """热力图可视化"""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def generate_metrics_heatmap(self,
                                 results: Dict[str, SceneEvaluationResult],
                                 output_path: Path) -> Path:
        """
        生成指标热力图

        Args:
            results: 所有场景的评价结果
            output_path: 输出路径

        Returns:
            Path: 保存的图片路径
        """
        logger.info("生成指标热力图...")

        # 构建数据矩阵
        scene_names = sorted(results.keys())
        all_metrics = set()

        for result in results.values():
            metrics_dict = result.get_metrics_dict()
            all_metrics.update(metrics_dict.keys())

        all_metrics = sorted(list(all_metrics))

        if not all_metrics:
            logger.warning("无有效的指标数据")
            return None

        # 创建矩阵
        data_matrix = np.zeros((len(all_metrics), len(scene_names)))

        for j, scene_name in enumerate(scene_names):
            result = results[scene_name]
            metrics_dict = result.get_metrics_dict()
            for i, metric_name in enumerate(all_metrics):
                data_matrix[i, j] = metrics_dict.get(metric_name, 0)

        # 创建热力图
        fig, ax = plt.subplots(figsize=(14, 10))

        # 简化指标名称显示
        metric_labels = [name.replace('_', '\n') for name in all_metrics]

        sns.heatmap(data_matrix,
                    xticklabels=scene_names,
                    yticklabels=metric_labels,
                    annot=True,
                    fmt='.1f',
                    cmap='RdYlGn',
                    center=50,
                    vmin=0,
                    vmax=100,
                    cbar_kws={'label': '评分'},
                    ax=ax,
                    linewidths=0.5)

        plt.title('评估指标热力图', size=14, weight='bold', pad=20, fontproperties=chinese_font_prop)
        plt.xlabel('场景', size=12, fontproperties=chinese_font_prop)
        plt.ylabel('指标', size=12, fontproperties=chinese_font_prop)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        # 保存
        output_file = output_path / f's2_metrics_heatmap.{self.config.save_format}'
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"热力图已保存: {output_file}")
        return output_file

    def generate_category_heatmap(self,
                                  results: Dict[str, SceneEvaluationResult],
                                  output_path: Path) -> Path:
        """
        生成按类别的热力图

        Args:
            results: 所有场景的评价结果
            output_path: 输出路径

        Returns:
            Path: 保存的图片路径
        """
        logger.info("生成分类指标热力图...")

        # 收集各类别的指标
        scene_names = sorted(results.keys())
        categories = ['data_quality', 'report_quality', 'end_to_end', 'comprehensive']
        category_labels = ['数据质量', '报告质量', '端到端性能', '综合评分']

        data_matrix = np.zeros((len(categories), len(scene_names)))

        for j, scene_name in enumerate(scene_names):
            result = results[scene_name]
            for i, category in enumerate(categories):
                if category in result.metrics_results:
                    values = [m.metric_value for m in result.metrics_results[category].values()]
                    data_matrix[i, j] = np.mean(values) if values else 0

        # 创建热力图
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.heatmap(data_matrix,
                    xticklabels=scene_names,
                    yticklabels=category_labels,
                    annot=True,
                    fmt='.1f',
                    cmap='YlGnBu',
                    vmin=0,
                    vmax=100,
                    cbar_kws={'label': '评分'},
                    ax=ax,
                    linewidths=1)

        plt.title('评估类别热力图', size=14, weight='bold', pad=20, fontproperties=chinese_font_prop)
        plt.xlabel('场景', size=12, fontproperties=chinese_font_prop)
        plt.ylabel('评估类别', size=12, fontproperties=chinese_font_prop)
        plt.xticks(rotation=45, ha='right')
        
        # 设置轴标签字体
        ax.set_xticklabels(scene_names, fontproperties=chinese_font_prop)
        ax.set_yticklabels(category_labels, fontproperties=chinese_font_prop)

        # 保存
        output_file = output_path / f's2_metrics_heatmap_category.{self.config.save_format}'
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"分类热力图已保存: {output_file}")
        return output_file


class BarChartVisualizer:
    """柱状图可视化"""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def generate_overall_scores_bar(self,
                                    results: Dict[str, SceneEvaluationResult],
                                    output_path: Path) -> Path:
        """
        生成综合评分柱状图

        Args:
            results: 所有场景的评价结果
            output_path: 输出路径

        Returns:
            Path: 保存的图片路径
        """
        logger.info("生成综合评分柱状图...")

        scene_names = sorted(results.keys())
        scores = [results[name].overall_score for name in scene_names]

        fig, ax = plt.subplots(figsize=self.config.figure_size)

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(scores)))
        bars = ax.bar(range(len(scene_names)), scores, color=colors, edgecolor='black', linewidth=1.5)

        # 在柱子上添加数值标签
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{score:.1f}',
                    ha='center', va='bottom', fontsize=10, weight='bold')

        ax.set_xlabel('场景', fontsize=12, weight='bold', fontproperties=chinese_font_prop)
        ax.set_ylabel('综合评分', fontsize=12, weight='bold', fontproperties=chinese_font_prop)
        ax.set_title('各场景综合评分对比', fontsize=14, weight='bold', pad=20, fontproperties=chinese_font_prop)
        ax.set_xticks(range(len(scene_names)))
        ax.set_xticklabels(scene_names, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='中等评分线')
        ax.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='优秀评分线')
        ax.legend(prop=chinese_font_prop)
        ax.grid(axis='y', alpha=0.3)

        # 保存
        output_file = output_path / f's3_metrics_bars_overall.{self.config.save_format}'
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"综合评分柱状图已保存: {output_file}")
        return output_file

    def generate_category_comparison_bar(self,
                                         results: Dict[str, SceneEvaluationResult],
                                         output_path: Path) -> Path:
        """
        生成类别评分对比柱状图

        Args:
            results: 所有场景的评价结果
            output_path: 输出路径

        Returns:
            Path: 保存的图片路径
        """
        logger.info("生成类别对比柱状图...")

        scene_names = sorted(results.keys())
        categories = ['data_quality', 'report_quality', 'end_to_end']
        category_labels = ['数据质量', '报告质量', '端到端性能']

        # 收集数据
        category_scores = {cat: [] for cat in categories}

        for scene_name in scene_names:
            result = results[scene_name]
            for category in categories:
                if category in result.metrics_results:
                    values = [m.metric_value for m in result.metrics_results[category].values()]
                    category_scores[category].append(np.mean(values) if values else 0)
                else:
                    category_scores[category].append(0)

        # 创建柱状图
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        x = np.arange(len(scene_names))
        width = 0.25

        for i, (category, label) in enumerate(zip(categories, category_labels)):
            offset = (i - 1) * width
            ax.bar(x + offset, category_scores[category], width, label=label)

        ax.set_xlabel('场景', fontsize=12, weight='bold', fontproperties=chinese_font_prop)
        ax.set_ylabel('评分', fontsize=12, weight='bold', fontproperties=chinese_font_prop)
        ax.set_title('场景评估类别对比', fontsize=14, weight='bold', pad=20, fontproperties=chinese_font_prop)
        ax.set_xticks(x)
        ax.set_xticklabels(scene_names, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.legend(prop=chinese_font_prop)
        ax.grid(axis='y', alpha=0.3)

        # 保存
        output_file = output_path / f's3_metrics_bars_category.{self.config.save_format}'
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"类别对比柱状图已保存: {output_file}")
        return output_file


class LineChartVisualizer:
    """线性图可视化"""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def generate_metrics_distribution(self,
                                      results: Dict[str, SceneEvaluationResult],
                                      output_path: Path) -> Path:
        """
        生成指标分布线性图

        Args:
            results: 所有场景的评价结果
            output_path: 输出路径

        Returns:
            Path: 保存的图片路径
        """
        logger.info("生成指标分布线性图...")

        scene_names = sorted(results.keys())
        all_metrics = set()

        for result in results.values():
            metrics_dict = result.get_metrics_dict()
            all_metrics.update(metrics_dict.keys())

        all_metrics = sorted(list(all_metrics))[:6]  # 选择前6个指标以避免拥挤

        fig, ax = plt.subplots(figsize=self.config.figure_size)

        for metric in all_metrics:
            values = []
            for scene_name in scene_names:
                result = results[scene_name]
                metrics_dict = result.get_metrics_dict()
                values.append(metrics_dict.get(metric, 0))

            ax.plot(range(len(scene_names)), values, marker='o', label=metric, linewidth=2)

        ax.set_xlabel('场景', fontsize=12, weight='bold', fontproperties=chinese_font_prop)
        ax.set_ylabel('评分', fontsize=12, weight='bold', fontproperties=chinese_font_prop)
        ax.set_title('主要指标分布趋势', fontsize=14, weight='bold', pad=20, fontproperties=chinese_font_prop)
        ax.set_xticks(range(len(scene_names)))
        ax.set_xticklabels(scene_names, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # 保存
        output_file = output_path / f's4_metrics_distribution.{self.config.save_format}'
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"指标分布线性图已保存: {output_file}")
        return output_file


class ScatterPlotVisualizer:
    """散点图可视化"""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def generate_metrics_correlation(self,
                                     results: Dict[str, SceneEvaluationResult],
                                     output_path: Path) -> Path:
        """
        生成指标相关性散点图

        Args:
            results: 所有场景的评价结果
            output_path: 输出路径

        Returns:
            Path: 保存的图片路径
        """
        logger.info("生成指标相关性散点图...")

        # 收集所有指标
        all_metrics = set()
        metrics_data = {}

        for result in results.values():
            metrics_dict = result.get_metrics_dict()
            all_metrics.update(metrics_dict.keys())
            for metric_name, value in metrics_dict.items():
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = []
                metrics_data[metric_name].append(value)

        all_metrics = sorted(list(all_metrics))

        if len(all_metrics) < 2:
            logger.warning("指标数量不足，无法生成相关性图")
            return None

        # 创建子图
        num_metrics = min(len(all_metrics), 6)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx in range(num_metrics - 1):
            ax = axes[idx]
            metric1 = all_metrics[idx]
            metric2 = all_metrics[idx + 1]

            x = metrics_data[metric1]
            y = metrics_data[metric2]

            ax.scatter(x, y, s=100, alpha=0.6, edgecolors='k')

            # 计算相关系数
            correlation = np.corrcoef(x, y)[0, 1]

            ax.set_xlabel(metric1.replace('_', '\n'), fontsize=9)
            ax.set_ylabel(metric2.replace('_', '\n'), fontsize=9)
            ax.set_title(f'相关系数: {correlation:.3f}', fontsize=10, fontproperties=chinese_font_prop)
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(num_metrics - 1, 6):
            axes[idx].set_visible(False)

        plt.suptitle('评估指标相关性分析', fontsize=14, weight='bold', y=1.00, fontproperties=chinese_font_prop)

        # 保存
        output_file = output_path / f's5_metrics_correlation.{self.config.save_format}'
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"相关性散点图已保存: {output_file}")
        return output_file


class BoxPlotVisualizer:
    """箱线图可视化"""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def generate_metrics_distribution_box(self,
                                          results: Dict[str, SceneEvaluationResult],
                                          output_path: Path) -> Path:
        """
        生成指标分布箱线图

        Args:
            results: 所有场景的评价结果
            output_path: 输出路径

        Returns:
            Path: 保存的图片路径
        """
        logger.info("生成指标分布箱线图...")

        # 收集指标数据
        categories = ['data_quality', 'report_quality', 'end_to_end']
        category_labels = ['数据质量', '报告质量', '端到端性能']
        category_data = {cat: [] for cat in categories}

        for result in results.values():
            for category in categories:
                if category in result.metrics_results:
                    values = [m.metric_value for m in result.metrics_results[category].values()]
                    category_data[category].extend(values)

        # 创建箱线图
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        data_to_plot = [category_data[cat] for cat in categories]

        bp = ax.boxplot(data_to_plot, labels=category_labels, patch_artist=True)

        # 设置颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel('评分', fontsize=12, weight='bold', fontproperties=chinese_font_prop)
        ax.set_title('评估指标分布箱线图', fontsize=14, weight='bold', pad=20, fontproperties=chinese_font_prop)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        
        # 设置X轴标签字体
        ax.set_xticklabels(category_labels, fontproperties=chinese_font_prop)

        # 保存
        output_file = output_path / f's6_metrics_boxplot.{self.config.save_format}'
        plt.tight_layout()
        plt.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"箱线图已保存: {output_file}")
        return output_file


class ComparisonVisualizer:
    """综合对比可视化"""

    def __init__(self, config: VisualizationConfig):
        self.config = config

    def generate_overall_comparison(self,
                                    results: Dict[str, SceneEvaluationResult],
                                    summary: Dict,
                                    output_path: Path) -> Path:
        """
        生成综合对比图表

        Args:
            results: 所有场景的评价结果
            summary: 评价摘要
            output_path: 输出路径

        Returns:
            Path: 保存的图片路径
        """
        logger.info("生成综合对比图表...")

        scene_names = sorted(results.keys())
        scores = [results[name].overall_score for name in scene_names]
        avg_score = summary['average_score']

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. 综合评分柱状图
        ax1 = fig.add_subplot(gs[0, 0])
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(scores)))
        bars = ax1.bar(range(len(scene_names)), scores, color=colors, edgecolor='black')
        ax1.axhline(y=avg_score, color='r', linestyle='--', label=f'平均分: {avg_score:.1f}')
        ax1.set_xlabel('场景', fontproperties=chinese_font_prop)
        ax1.set_ylabel('评分', fontproperties=chinese_font_prop)
        ax1.set_title('综合评分对比', fontproperties=chinese_font_prop)
        ax1.set_xticks(range(len(scene_names)))
        ax1.set_xticklabels(scene_names, rotation=45, ha='right')
        ax1.set_ylim(0, 105)
        ax1.legend(prop=chinese_font_prop)
        ax1.grid(axis='y', alpha=0.3)

        # 2. 成功/失败统计饼图
        ax2 = fig.add_subplot(gs[0, 1])
        success_count = summary['successful_scenes']
        failed_count = summary['failed_scenes']
        sizes = [success_count, failed_count]
        labels = [f'成功\n({success_count})', f'失败\n({failed_count})']
        colors_pie = ['#90EE90', '#FFB6C6']
        explode = (0.05, 0)
        ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                shadow=True, startangle=90, textprops={'fontproperties': chinese_font_prop})
        ax2.set_title('评价结果统计', fontproperties=chinese_font_prop)

        # 3. 评分分布直方图
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(scores, bins=max(3, len(scores) // 2), color='skyblue', edgecolor='black')
        ax3.axvline(x=avg_score, color='r', linestyle='--', linewidth=2, label=f'平均: {avg_score:.1f}')
        ax3.set_xlabel('评分', fontproperties=chinese_font_prop)
        ax3.set_ylabel('场景数', fontproperties=chinese_font_prop)
        ax3.set_title('评分分布', fontproperties=chinese_font_prop)
        ax3.legend(prop=chinese_font_prop)
        ax3.grid(axis='y', alpha=0.3)

        # 4. 统计信息文本框
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        stats_text = f"""
        评价统计摘要
        ━━━━━━━━━━━━━━━━━━━━

        总场景数:          {summary['total_scenes']}
        成功场景:          {success_count}
        失败场景:          {failed_count}
        成功率:            {success_count / summary['total_scenes'] * 100:.1f}%

        综合评分
        ━━━━━━━━━━━━━━━━━━━━
        平均分:            {avg_score:.2f}
        最高分:            {max(scores):.2f}
        最低分:            {min(scores):.2f}
        标准差:            {np.std(scores):.2f}

        评价时间:          {summary['evaluation_time'][:10]}
        """

        font_prop = chinese_font_prop if chinese_font_prop else fm.FontProperties(family='monospace')
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                 fontsize=11, verticalalignment='top', fontproperties=font_prop,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('评估结果综合对比', fontsize=16, weight='bold', y=0.995, fontproperties=chinese_font_prop)

        # 保存
        output_file = output_path / f's7_overall_comparison.{self.config.save_format}'
        plt.savefig(output_file, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"综合对比图表已保存: {output_file}")
        return output_file


class DashboardGenerator:
    """仪表板生成器"""

    def __init__(self, config: VisualizationConfig, output_path: Path):
        self.config = config
        self.output_path = output_path
        self.figures_path = output_path / 'figures'
        self.figures_path.mkdir(parents=True, exist_ok=True)

    def generate_html_dashboard(self,
                                results: Dict[str, SceneEvaluationResult],
                                summary: Dict) -> Path:
        """
        生成HTML仪表板

        Args:
            results: 所有场景的评价结果
            summary: 评价摘要

        Returns:
            Path: HTML文件路径
        """
        logger.info("生成HTML仪表板...")

        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>交通场景理解系统评价仪表板</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }
                .container {
                    max-width: 1400px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                    padding: 30px;
                }
                .header {
                    text-align: center;
                    border-bottom: 3px solid #667eea;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }
                .header h1 {
                    color: #333;
                    font-size: 32px;
                    margin-bottom: 10px;
                }
                .header p {
                    color: #666;
                    font-size: 14px;
                }
                .summary-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }
                .summary-card {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }
                .summary-card h3 {
                    font-size: 12px;
                    text-transform: uppercase;
                    margin-bottom: 10px;
                    opacity: 0.9;
                }
                .summary-card .value {
                    font-size: 28px;
                    font-weight: bold;
                }
                .scenes-section {
                    margin-bottom: 30px;
                }
                .scenes-section h2 {
                    color: #333;
                    font-size: 20px;
                    margin-bottom: 15px;
                    border-left: 4px solid #667eea;
                    padding-left: 10px;
                }
                .scenes-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                .scenes-table th {
                    background: #667eea;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }
                .scenes-table td {
                    padding: 12px;
                    border-bottom: 1px solid #ddd;
                }
                .scenes-table tr:hover {
                    background: #f5f5f5;
                }
                .score-bar {
                    width: 100%;
                    height: 24px;
                    background: #e0e0e0;
                    border-radius: 4px;
                    overflow: hidden;
                    position: relative;
                }
                .score-fill {
                    height: 100%;
                    background: linear-gradient(90deg, #ff6b6b 0%, #ffd93d 50%, #6bcf7f 100%);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    font-size: 12px;
                }
                .status-success {
                    color: #10b981;
                    font-weight: bold;
                }
                .status-failed {
                    color: #ef4444;
                    font-weight: bold;
                }
                .figures-section {
                    margin-top: 30px;
                }
                .figures-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                .figure-card {
                    background: #f9f9f9;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                .figure-card img {
                    width: 100%;
                    height: auto;
                    display: block;
                }
                .footer {
                    text-align: center;
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    color: #999;
                    font-size: 12px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>交通场景理解系统评价仪表板</h1>
                    <p>三阶段解耦架构的可信交通场景理解系统 - 综合评估报告</p>
                </div>
        """

        # 添加摘要卡片
        html_content += """
                <div class="summary-grid">
        """

        html_content += f"""
                    <div class="summary-card">
                        <h3>总场景数</h3>
                        <div class="value">{summary['total_scenes']}</div>
                    </div>
                    <div class="summary-card">
                        <h3>成功场景</h3>
                        <div class="value">{summary['successful_scenes']}</div>
                    </div>
                    <div class="summary-card">
                        <h3>平均评分</h3>
                        <div class="value">{summary['average_score']:.2f}</div>
                    </div>
                    <div class="summary-card">
                        <h3>成功率</h3>
                        <div class="value">{summary['successful_scenes'] / summary['total_scenes'] * 100:.1f}%</div>
                    </div>
        """

        html_content += """
                </div>
        """

        # 添加场景表格
        html_content += """
                <div class="scenes-section">
                    <h2>场景评价结果</h2>
                    <table class="scenes-table">
                        <thead>
                            <tr>
                                <th>场景名称</th>
                                <th>状态</th>
                                <th>综合评分</th>
                                <th>评分可视化</th>
                            </tr>
                        </thead>
                        <tbody>
        """

        for scene_name in sorted(results.keys()):
            result = results[scene_name]
            status = '✓ 成功' if result.status == 'success' else '✗ 失败'
            status_class = 'status-success' if result.status == 'success' else 'status-failed'
            score = result.overall_score

            html_content += f"""
                            <tr>
                                <td>{scene_name}</td>
                                <td><span class="{status_class}">{status}</span></td>
                                <td>{score:.2f}</td>
                                <td>
                                    <div class="score-bar">
                                        <div class="score-fill" style="width: {score}%">{score:.1f}</div>
                                    </div>
                                </td>
                            </tr>
            """

        html_content += """
                        </tbody>
                    </table>
                </div>
        """

        # 添加图表部分
        html_content += """
                <div class="figures-section">
                    <h2>评估图表</h2>
                    <div class="figures-grid">
        """

        # 列出所有生成的图表
        for figure_file in sorted(self.figures_path.glob(f'*.{self.config.save_format}')):
            rel_path = figure_file.relative_to(self.output_path)
            html_content += f"""
                        <div class="figure-card">
                            <img src="{rel_path}" alt="{figure_file.stem}">
                        </div>
            """

        html_content += """
                    </div>
                </div>
        """

        # 添加页脚
        html_content += f"""
                <div class="footer">
                    <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>交通场景理解系统评价框架 v1.0</p>
                </div>
            </div>
        </body>
        </html>
        """

        output_file = self.output_path / 'dashboard.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML仪表板已生成: {output_file}")
        return output_file


class VisualizationManager:
    """可视化管理器 - 协调所有可视化模块"""

    def __init__(self, runner: EvaluationRunner, config: VisualizationConfig = None):
        """
        初始化可视化管理器

        Args:
            runner: 评价运行引擎
            config: 可视化配置
        """
        self.runner = runner
        self.config = config or VisualizationConfig()
        self.output_path = runner.config.output_path
        self.figures_path = self.output_path / 'figures'
        self.figures_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"可视化管理器初始化，输出路径: {self.figures_path}")

    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        logger.info("=" * 80)
        logger.info("开始生成所有可视化图表".center(80))
        logger.info("=" * 80)

        results = self.runner.results
        summary = self.runner.summary

        if not results:
            logger.warning("没有评价结果，无法生成可视化")
            return

        # 1. 雷达图
        radar = RadarChartVisualizer(self.config)
        for scene_name, result in results.items():
            radar.generate_scene_radar(scene_name, result, self.figures_path)
        radar.generate_multi_scene_radar(results, self.figures_path)

        # 2. 热力图
        heatmap = HeatmapVisualizer(self.config)
        heatmap.generate_metrics_heatmap(results, self.figures_path)
        heatmap.generate_category_heatmap(results, self.figures_path)

        # 3. 柱状图
        bar = BarChartVisualizer(self.config)
        bar.generate_overall_scores_bar(results, self.figures_path)
        bar.generate_category_comparison_bar(results, self.figures_path)

        # 4. 线性图
        line = LineChartVisualizer(self.config)
        line.generate_metrics_distribution(results, self.figures_path)

        # 5. 散点图
        scatter = ScatterPlotVisualizer(self.config)
        scatter.generate_metrics_correlation(results, self.figures_path)

        # 6. 箱线图
        boxplot = BoxPlotVisualizer(self.config)
        boxplot.generate_metrics_distribution_box(results, self.figures_path)

        # 7. 综合对比
        comparison = ComparisonVisualizer(self.config)
        comparison.generate_overall_comparison(results, summary, self.figures_path)

        # 8. HTML仪表板
        dashboard = DashboardGenerator(self.config, self.output_path)
        dashboard.generate_html_dashboard(results, summary)

        logger.info("所有可视化图表生成完成")


if __name__ == "__main__":
    # 示例使用
    config = EvaluationConfig(
        data_path='./data',
        output_path='./output'
    )

    runner = EvaluationRunner(config)
    runner.run()

    # 生成可视化
    viz_manager = VisualizationManager(runner)
    viz_manager.generate_all_visualizations()
