# 评价框架配置模块

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvaluationConfig:
    """评价配置"""
    data_path: Path = Path('./data')
    output_path: Path = Path('./output')
    report_format: str = 'json'
    enable_visualization: bool = True


@dataclass
class VisualizationConfig:
    """可视化配置"""
    figure_dpi: int = 300
    figure_size: tuple = (12, 8)
    color_palette: str = 'husl'
    style: str = 'seaborn-v0_8-darkgrid'
    save_format: str = 'png'


@dataclass
class ReportConfig:
    """报告配置"""
    title: str = "交通场景理解系统评价报告"
    author: str = "智慧交通队"
    version: str = "1.0"
    include_detailed_metrics: bool = True
    include_recommendations: bool = True
    language: str = 'zh_CN'


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_path: str = 'evaluation.log'


DEFAULT_EVAL_CONFIG = EvaluationConfig()
DEFAULT_VIZ_CONFIG = VisualizationConfig()
DEFAULT_REPORT_CONFIG = ReportConfig()
DEFAULT_LOGGING_CONFIG = LoggingConfig()
