"""
评价运行引擎模块
负责协调数据加载、指标计算、结果管理和报告生成
"""

import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import logging
import traceback
from dataclasses import asdict

from .metrics import (
    MetricsEvaluator,
    MetricResult,
    DataIntegrityMetric,
    DataConsistencyMetric,
    GraphComplexityMetric,
    NodeAttributeRichnessMetric,
    ReportCompletionMetric,
    InformationDensityMetric,
    ReportReadabilityMetric,
    EventDetectionMetric,
    Data2ReportAlignmentMetric,
    SystemComprehensiveScore
)
from .data_loader import SmartDataLoader
from .config import EvaluationConfig

logger = logging.getLogger(__name__)


class SceneEvaluationResult:
    """单个场景的评价结果"""

    def __init__(self, scene_name: str):
        self.scene_name = scene_name
        self.evaluation_time = datetime.now()
        self.metrics_results = {}  # Dict[str, Dict[str, MetricResult]]
        self.overall_score = 0.0
        self.status = 'pending'  # pending, success, failed
        self.error_message = None
        self.load_status = {}

    def add_metrics(self, category: str, metrics: Dict[str, MetricResult]):
        """添加指标结果"""
        self.metrics_results[category] = metrics

    def set_overall_score(self, score: float):
        """设置综合评分"""
        self.overall_score = score

    def set_success(self):
        """标记为成功"""
        self.status = 'success'

    def set_failed(self, error_message: str):
        """标记为失败"""
        self.status = 'failed'
        self.error_message = error_message

    def to_dict(self) -> Dict:
        """转换为字典"""
        result = {
            'scene_name': self.scene_name,
            'evaluation_time': self.evaluation_time.isoformat(),
            'status': self.status,
            'overall_score': self.overall_score,
            'metrics': {}
        }

        for category, metrics in self.metrics_results.items():
            result['metrics'][category] = {}
            for metric_name, metric_result in metrics.items():
                result['metrics'][category][metric_name] = asdict(metric_result)

        if self.error_message:
            result['error_message'] = self.error_message

        return result

    def get_metrics_dict(self) -> Dict[str, float]:
        """获取所有指标的值字典"""
        metrics_dict = {}
        for category, metrics in self.metrics_results.items():
            for metric_name, metric_result in metrics.items():
                metrics_dict[metric_name] = metric_result.metric_value
        return metrics_dict


class EvaluationRunner:
    """评价运行引擎"""

    def __init__(self, config: EvaluationConfig):
        """
        初始化评价引擎

        Args:
            config: 评价配置
        """
        self.config = config
        self.data_loader = SmartDataLoader(str(config.data_path))
        self.results = {}  # Dict[str, SceneEvaluationResult]
        self.summary = {}

        logger.info(f"评价引擎初始化成功: {config}")

    def load_data(self, scene_names: List[str] = None) -> bool:
        """
        加载数据

        Args:
            scene_names: 场景名称列表，为None则加载所有

        Returns:
            bool: 是否加载成功
        """
        try:
            logger.info("开始加载数据...")
            prepared_data = self.data_loader.load_and_prepare(scene_names)
            logger.info(f"成功加载 {len(prepared_data)} 个场景的数据")
            return True
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def evaluate_single_scene(self, scene_name: str) -> SceneEvaluationResult:
        """
        评价单个场景

        Args:
            scene_name: 场景名称

        Returns:
            SceneEvaluationResult: 场景评价结果
        """
        logger.info(f"开始评价场景: {scene_name}")

        result = SceneEvaluationResult(scene_name)

        try:
            # 获取场景数据
            graph_data, report_text = self.data_loader.get_scene_pair(scene_name)

            if graph_data is None and report_text is None:
                raise ValueError(f"无法加载场景 {scene_name} 的数据")

            result.load_status = {
                'has_graph': graph_data is not None,
                'has_report': report_text is not None
            }

            # 创建评估器
            evaluator = MetricsEvaluator(graph_data or {}, report_text or "")

            # 评价数据质量
            if graph_data:
                logger.info(f"  评价数据质量...")
                data_quality_metrics = evaluator.evaluate_data_quality()
                result.add_metrics('data_quality', data_quality_metrics)

            # 评价报告质量
            if report_text:
                logger.info(f"  评价报告质量...")
                report_quality_metrics = evaluator.evaluate_report_quality()
                result.add_metrics('report_quality', report_quality_metrics)

            # 评价端到端指标
            if graph_data and report_text:
                logger.info(f"  评价端到端指标...")
                end_to_end_metrics = evaluator.evaluate_end_to_end()
                result.add_metrics('end_to_end', end_to_end_metrics)

            # 评价综合评分
            all_metrics_dict = result.get_metrics_dict()
            if all_metrics_dict:
                comprehensive_calc = SystemComprehensiveScore(all_metrics_dict)
                comprehensive_score = comprehensive_calc.calculate()
                result.add_metrics('comprehensive', {'System_Comprehensive_Score': comprehensive_score})
                result.set_overall_score(comprehensive_score.metric_value)

            result.set_success()
            logger.info(f"场景 {scene_name} 评价完成，综合评分: {result.overall_score:.2f}")

        except Exception as e:
            error_msg = f"评价失败: {str(e)}"
            logger.error(f"场景 {scene_name} {error_msg}")
            logger.error(traceback.format_exc())
            result.set_failed(error_msg)

        return result

    def evaluate_all_scenes(self) -> Dict[str, SceneEvaluationResult]:
        """
        评价所有加载的场景

        Returns:
            Dict: 所有场景的评价结果
        """
        logger.info("=" * 80)
        logger.info("开始全面评价所有场景".center(80))
        logger.info("=" * 80)

        scene_pairs = self.data_loader.get_all_scene_pairs()

        if not scene_pairs:
            logger.warning("未找到任何场景数据")
            return {}

        logger.info(f"发现 {len(scene_pairs)} 个场景，开始逐个评价...")

        for scene_name, _, _ in scene_pairs:
            result = self.evaluate_single_scene(scene_name)
            self.results[scene_name] = result

        logger.info(f"完成 {len(self.results)} 个场景的评价")

        return self.results

    def generate_summary(self) -> Dict:
        """
        生成评价摘要

        Returns:
            Dict: 评价摘要
        """
        logger.info("生成评价摘要...")

        summary = {
            'evaluation_time': datetime.now().isoformat(),
            'total_scenes': len(self.results),
            'successful_scenes': 0,
            'failed_scenes': 0,
            'average_score': 0.0,
            'metrics_statistics': {},
            'scene_results': {}
        }

        scores = []
        all_metrics_by_name = {}

        # 统计结果
        for scene_name, result in self.results.items():
            if result.status == 'success':
                summary['successful_scenes'] += 1
                scores.append(result.overall_score)
            else:
                summary['failed_scenes'] += 1

            summary['scene_results'][scene_name] = {
                'status': result.status,
                'overall_score': result.overall_score,
                'error_message': result.error_message
            }

            # 收集指标
            metrics_dict = result.get_metrics_dict()
            for metric_name, metric_value in metrics_dict.items():
                if metric_name not in all_metrics_by_name:
                    all_metrics_by_name[metric_name] = []
                all_metrics_by_name[metric_name].append(metric_value)

        # 计算平均评分
        if scores:
            summary['average_score'] = sum(scores) / len(scores)

        # 计算指标统计
        for metric_name, values in all_metrics_by_name.items():
            summary['metrics_statistics'][metric_name] = {
                'mean': float(sum(values) / len(values)),
                'min': float(min(values)),
                'max': float(max(values)),
                'count': len(values)
            }

        self.summary = summary
        logger.info("评价摘要生成完成")

        return summary

    def save_results(self, format_type: str = 'json'):
        """
        保存评价结果

        Args:
            format_type: 保存格式 ('json' 或 'csv')
        """
        logger.info(f"保存评价结果为 {format_type} 格式...")

        if format_type == 'json' or format_type == 'both':
            self._save_json_results()

        if format_type == 'csv' or format_type == 'both':
            self._save_csv_results()

        logger.info("评价结果保存完成")

    def _save_json_results(self):
        """保存为JSON格式"""
        # 保存详细结果
        detailed_results = {
            'summary': self.summary,
            'scene_evaluations': {}
        }

        for scene_name, result in self.results.items():
            detailed_results['scene_evaluations'][scene_name] = result.to_dict()

        output_file = self.config.output_path / 'evaluation_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)

        logger.info(f"详细结果已保存: {output_file}")

        # 保存摘要
        summary_file = self.config.output_path / 'metrics_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.summary, f, ensure_ascii=False, indent=2)

        logger.info(f"摘要已保存: {summary_file}")

    def _save_csv_results(self):
        """保存为CSV格式"""
        import csv

        # 保存场景级别的结果
        scenes_csv_file = self.config.output_path / 'scenes_results.csv'

        with open(scenes_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Scene', 'Status', 'Overall_Score', 'Error_Message'])

            for scene_name, result in self.results.items():
                writer.writerow([
                    scene_name,
                    result.status,
                    result.overall_score,
                    result.error_message or ''
                ])

        logger.info(f"场景结果已保存: {scenes_csv_file}")

        # 保存指标统计
        metrics_csv_file = self.config.output_path / 'metrics_summary.csv'

        with open(metrics_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric_Name', 'Mean', 'Min', 'Max', 'Count'])

            for metric_name, stats in self.summary['metrics_statistics'].items():
                writer.writerow([
                    metric_name,
                    f"{stats['mean']:.2f}",
                    f"{stats['min']:.2f}",
                    f"{stats['max']:.2f}",
                    stats['count']
                ])

        logger.info(f"指标统计已保存: {metrics_csv_file}")

    def print_summary_report(self):
        """打印评价摘要报告"""
        print("\n" + "=" * 100)
        print("交通场景理解系统评价总结报告".center(100))
        print("=" * 100 + "\n")

        print(f"评价时间: {self.summary['evaluation_time']}")
        print(f"总场景数: {self.summary['total_scenes']}")
        print(f"成功评价: {self.summary['successful_scenes']}")
        print(f"评价失败: {self.summary['failed_scenes']}")
        print(f"平均评分: {self.summary['average_score']:.2f}\n")

        print("-" * 100)
        print("【场景评价结果】")
        print("-" * 100)

        for scene_name, result_info in self.summary['scene_results'].items():
            status_icon = "[OK]" if result_info['status'] == 'success' else "[FAIL]"
            print(f"{status_icon} {scene_name}: {result_info['overall_score']:.2f}")
            if result_info['error_message']:
                print(f"    错误: {result_info['error_message']}")

        print("\n" + "-" * 100)
        print("【指标统计汇总】")
        print("-" * 100)

        for metric_name, stats in sorted(self.summary['metrics_statistics'].items()):
            print(f"{metric_name}:")
            print(f"  平均值: {stats['mean']:.2f}")
            print(f"  最小值: {stats['min']:.2f}")
            print(f"  最大值: {stats['max']:.2f}")
            print(f"  样本数: {stats['count']}")

        print("\n" + "=" * 100)

    def print_detailed_report(self):
        """打印详细评价报告"""
        print("\n" + "=" * 100)
        print("交通场景理解系统详细评价报告".center(100))
        print("=" * 100 + "\n")

        for scene_name, result in self.results.items():
            print(f"\n【场景: {scene_name}】")
            print("-" * 100)

            if result.status == 'failed':
                print(f"[FAIL] 评价失败: {result.error_message}")
                continue

            print(f"[OK] 综合评分: {result.overall_score:.2f}\n")

            # 打印各类指标
            for category, metrics in result.metrics_results.items():
                print(f"\n  【{category.upper()}】")
                for metric_name, metric_result in metrics.items():
                    print(f"    {metric_name}: {metric_result.metric_value} {metric_result.unit}")
                    print(f"      公式: {metric_result.formula}")
                    print(f"      说明: {metric_result.description}")

        print("\n" + "=" * 100)

    def run(self, scene_names: List[str] = None, print_detailed: bool = False):
        """
        运行完整的评价流程

        Args:
            scene_names: 场景名称列表
            print_detailed: 是否打印详细报告
        """
        # 加载数据
        if not self.load_data(scene_names):
            logger.error("数据加载失败，中止评价")
            return False

        # 评价所有场景
        self.evaluate_all_scenes()

        # 生成摘要
        self.generate_summary()

        # 保存结果
        self.save_results(self.config.report_format)

        # 打印报告
        self.print_summary_report()

        if print_detailed:
            self.print_detailed_report()

        logger.info("评价流程完成")

        return True

    def get_results_for_visualization(self) -> Dict:
        """
        获取用于可视化的数据

        Returns:
            Dict: 可视化数据
        """
        viz_data = {
            'scene_names': list(self.results.keys()),
            'overall_scores': [result.overall_score for result in self.results.values()],
            'metrics_by_category': {},
            'all_metrics': {}
        }

        # 按类别整理指标
        categories = set()
        for result in self.results.values():
            categories.update(result.metrics_results.keys())

        for category in categories:
            viz_data['metrics_by_category'][category] = {}

            # 收集该类别的所有指标
            for metric_name in set():
                for result in self.results.values():
                    if category in result.metrics_results:
                        for name in result.metrics_results[category].keys():
                            set().add(name)

            # 为每个指标创建值列表
            for result in self.results.values():
                if category in result.metrics_results:
                    for metric_name, metric_result in result.metrics_results[category].items():
                        if metric_name not in viz_data['metrics_by_category'][category]:
                            viz_data['metrics_by_category'][category][metric_name] = []
                        viz_data['metrics_by_category'][category][metric_name].append(
                            metric_result.metric_value
                        )

        # 整理所有指标
        for result in self.results.values():
            metrics_dict = result.get_metrics_dict()
            for metric_name, value in metrics_dict.items():
                if metric_name not in viz_data['all_metrics']:
                    viz_data['all_metrics'][metric_name] = []
                viz_data['all_metrics'][metric_name].append(value)

        return viz_data


if __name__ == "__main__":
    # 示例使用
    config = EvaluationConfig(
        data_path='./data',
        output_path='./output',
        report_format='json',
        enable_visualization=True
    )

    runner = EvaluationRunner(config)
    runner.run(print_detailed=False)
