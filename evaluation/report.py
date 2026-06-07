"""
报告生成模块
生成详细的PDF和HTML评价报告
"""

import json
import os
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import logging
from jinja2 import Template
import numpy as np

from .runner import EvaluationRunner, SceneEvaluationResult
from .config import ReportConfig

logger = logging.getLogger(__name__)


class RecommendationGenerator:
    """建议生成器"""

    @staticmethod
    def generate_recommendations(result: SceneEvaluationResult) -> Dict[str, List[str]]:
        """
        基于评价结果生成改进建议

        Args:
            result: 场景评价结果

        Returns:
            Dict: 按类别分组的建议
        """
        recommendations = {
            'data_quality': [],
            'report_quality': [],
            'end_to_end': [],
            'general': []
        }

        metrics_dict = result.get_metrics_dict()

        # 数据质量建议
        if 'Data_Integrity' in metrics_dict:
            if metrics_dict['Data_Integrity'] < 80:
                recommendations['data_quality'].append(
                    "数据完整性低于80%，建议检查是否存在缺失字段，完善数据收集流程"
                )

        if 'Data_Consistency' in metrics_dict:
            if metrics_dict['Data_Consistency'] < 90:
                recommendations['data_quality'].append(
                    "数据一致性存在问题，建议验证节点-边关系的有效性"
                )

        if 'Graph_Complexity' in metrics_dict:
            if metrics_dict['Graph_Complexity'] < 30:
                recommendations['data_quality'].append(
                    "图复杂度较低，可考虑增加节点和边的数量以丰富场景表示"
                )

        if 'Node_Attribute_Richness' in metrics_dict:
            if metrics_dict['Node_Attribute_Richness'] < 60:
                recommendations['data_quality'].append(
                    "节点属性不够丰富，建议为节点添加更多描述性属性"
                )

        # 报告质量建议
        if 'Report_Completion' in metrics_dict:
            if metrics_dict['Report_Completion'] < 80:
                recommendations['report_quality'].append(
                    "报告缺少必要的部分信息，建议完善报告结构，确保涵盖所有关键要素"
                )

        if 'Report_Readability' in metrics_dict:
            if metrics_dict['Report_Readability'] < 60:
                recommendations['report_quality'].append(
                    "报告可读性需要改进，建议优化句子长度和表述方式"
                )

        if 'Information_Density' in metrics_dict:
            if metrics_dict['Information_Density'] < 5:
                recommendations['report_quality'].append(
                    "信息密度较低，建议增加关键信息词汇的使用"
                )

        if 'Event_Detection_Richness' in metrics_dict:
            if metrics_dict['Event_Detection_Richness'] < 60:
                recommendations['report_quality'].append(
                    "事件检测不够全面，建议在报告中涵盖更多类型的交通事件"
                )

        # 端到端建议
        if 'Data_Report_Alignment' in metrics_dict:
            if metrics_dict['Data_Report_Alignment'] < 70:
                recommendations['end_to_end'].append(
                    "数据-报告对齐度较低，建议加强从结构化数据到报告的映射"
                )

        # 综合建议
        if result.overall_score < 50:
            recommendations['general'].append(
                "整体评分较低，建议全面检查系统的各个环节"
            )
        elif result.overall_score < 70:
            recommendations['general'].append(
                "整体评分中等，建议重点改进得分较低的领域"
            )
        else:
            recommendations['general'].append(
                "整体表现良好，继续保持当前的质量标准"
            )

        return recommendations


class HTMLReportGenerator:
    """HTML报告生成器"""

    def __init__(self, config: ReportConfig):
        self.config = config

    def generate_report(self,
                        runner: EvaluationRunner,
                        output_path: Path) -> Path:
        """
        生成HTML报告

        Args:
            runner: 评价运行引擎
            output_path: 输出路径

        Returns:
            Path: 报告文件路径
        """
        logger.info("生成HTML评价报告...")

        html_content = self._build_html(runner)

        output_file = output_path / 'evaluation_report.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML报告已生成: {output_file}")
        return output_file

    def _build_html(self, runner: EvaluationRunner) -> str:
        """构建HTML内容"""

        results = runner.results
        summary = runner.summary

        # HTML头部
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.config.title}</title>
    <style>
        {{
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: #f5f7fa;
            }}

            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}

            /* 页面分割 */
            .page-break {{
                page-break-after: always;
                border-bottom: 2px solid #ddd;
                margin: 40px 0;
                padding-bottom: 40px;
            }}

            /* 封面页 */
            .cover {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 100px 40px;
                text-align: center;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            }}

            .cover h1 {{
                font-size: 48px;
                margin-bottom: 20px;
                font-weight: bold;
            }}

            .cover .subtitle {{
                font-size: 24px;
                margin-bottom: 60px;
                opacity: 0.9;
            }}

            .cover-info {{
                text-align: left;
                font-size: 16px;
                line-height: 2;
            }}

            .cover-info p {{
                margin: 10px 0;
            }}

            .cover-info span {{
                font-weight: bold;
            }}

            /* 普通内容页 */
            .content {{
                padding: 40px;
            }}

            .section {{
                margin-bottom: 40px;
            }}

            .section h1 {{
                color: #667eea;
                font-size: 32px;
                border-bottom: 3px solid #667eea;
                padding-bottom: 15px;
                margin-bottom: 25px;
            }}

            .section h2 {{
                color: #764ba2;
                font-size: 24px;
                margin-top: 30px;
                margin-bottom: 15px;
                border-left: 4px solid #667eea;
                padding-left: 15px;
            }}

            .section h3 {{
                color: #555;
                font-size: 18px;
                margin-top: 20px;
                margin-bottom: 10px;
            }}

            p {{
                margin-bottom: 12px;
                text-align: justify;
            }}

            /* 统计卡片 */
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}

            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }}

            .stat-card h4 {{
                font-size: 14px;
                text-transform: uppercase;
                margin-bottom: 10px;
                opacity: 0.9;
            }}

            .stat-card .value {{
                font-size: 36px;
                font-weight: bold;
            }}

            .stat-card .unit {{
                font-size: 14px;
                margin-top: 5px;
                opacity: 0.8;
            }}

            /* 表格 */
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}

            table th {{
                background: #667eea;
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: bold;
            }}

            table td {{
                padding: 12px 15px;
                border-bottom: 1px solid #eee;
            }}

            table tr:hover {{
                background: #f9f9f9;
            }}

            table tr:last-child td {{
                border-bottom: 2px solid #667eea;
            }}

            /* 进度条 */
            .progress-bar {{
                width: 100%;
                height: 24px;
                background: #e0e0e0;
                border-radius: 4px;
                overflow: hidden;
                margin: 10px 0;
            }}

            .progress-fill {{
                height: 100%;
                background: linear-gradient(90deg, #ff6b6b 0%, #ffd93d 50%, #6bcf7f 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 12px;
            }}

            /* 列表 */
            ul, ol {{
                margin: 15px 0 15px 25px;
            }}

            li {{
                margin-bottom: 8px;
                line-height: 1.8;
            }}

            /* 提示框 */
            .alert {{
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
                border-left: 4px solid;
            }}

            .alert.success {{
                background: #d4edda;
                border-color: #28a745;
                color: #155724;
            }}

            .alert.warning {{
                background: #fff3cd;
                border-color: #ffc107;
                color: #856404;
            }}

            .alert.danger {{
                background: #f8d7da;
                border-color: #dc3545;
                color: #721c24;
            }}

            .alert.info {{
                background: #d1ecf1;
                border-color: #17a2b8;
                color: #0c5460;
            }}

            /* 页脚 */
            .footer {{
                background: #f9f9f9;
                padding: 20px 40px;
                border-top: 1px solid #ddd;
                text-align: center;
                color: #666;
                font-size: 12px;
            }}

            .footer p {{
                margin: 5px 0;
            }}

            /* 打印样式 */
            @media print {{
                body {{
                    background: white;
                }}
                .container {{
                    box-shadow: none;
                }}
                .page-break {{
                    page-break-after: always;
                }}
            }}

            /* 指标详情 */
            .metric-item {{
                background: #f9f9f9;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #667eea;
                border-radius: 4px;
            }}

            .metric-item .name {{
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
            }}

            .metric-item .value {{
                font-size: 18px;
                color: #667eea;
                margin: 5px 0;
            }}

            .metric-item .formula {{
                font-family: monospace;
                color: #666;
                font-size: 12px;
                margin: 5px 0;
                word-break: break-all;
            }}

            .metric-item .description {{
                color: #999;
                font-size: 12px;
                margin-top: 8px;
            }}

            /* 推荐框 */
            .recommendation {{
                background: #f0f4ff;
                border-left: 4px solid #667eea;
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
            }}

            .recommendation::before {{
                content: "💡 ";
                margin-right: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
"""

        # 封面页
        html += self._build_cover(runner)

        # 目录
        html += self._build_toc()

        # 执行摘要
        html += self._build_executive_summary(runner)

        # 评价方法
        html += self._build_methodology()

        # 综合结果
        html += self._build_overall_results(runner)

        # 场景详细评价
        html += self._build_scene_details(runner)

        # 指标分析
        if self.config.include_detailed_metrics:
            html += self._build_detailed_metrics(runner)

        # 改进建议
        if self.config.include_recommendations:
            html += self._build_recommendations(runner)

        # 结论与展望
        html += self._build_conclusion(runner)

        # 页脚
        html += self._build_footer()

        html += """
    </div>
</body>
</html>
        """

        return html

    def _build_cover(self, runner: EvaluationRunner) -> str:
        """构建封面页"""
        return f"""
        <div class="cover">
            <h1>{self.config.title}</h1>
            <div class="subtitle">综合评估报告</div>
            <div class="cover-info">
                <p><span>版本:</span> {self.config.version}</p>
                <p><span>作者:</span> {self.config.author}</p>
                <p><span>生成时间:</span> {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
                <p><span>评估对象:</span> 三阶段解耦架构的可信交通场景理解系统</p>
            </div>
        </div>
        <div class="page-break"></div>
        """

    def _build_toc(self) -> str:
        """构建目录"""
        return """
        <div class="content">
            <div class="section">
                <h1>目录</h1>
                <ol>
                    <li>执行摘要</li>
                    <li>评价方法论</li>
                    <li>综合评价结果</li>
                    <li>场景详细评价</li>
                    <li>指标详细分析</li>
                    <li>改进建议与推荐</li>
                    <li>结论与展望</li>
                </ol>
            </div>
        </div>
        <div class="page-break"></div>
        """

    def _build_executive_summary(self, runner: EvaluationRunner) -> str:
        """构建执行摘要"""
        summary = runner.summary
        results = runner.results

        html = """
        <div class="content">
            <div class="section">
                <h1>1. 执行摘要</h1>
                <p>本报告对交通场景理解系统的第一阶段结构化数据和第三阶段交通报告进行了全面的评估。
                评估采用了多维度的可量化指标体系，包括数据质量、报告质量和端到端性能等方面。</p>

                <h2>关键发现</h2>
        """

        html += '<div class="stats-grid">'
        html += f"""
                <div class="stat-card">
                    <h4>总场景数</h4>
                    <div class="value">{summary['total_scenes']}</div>
                </div>
                <div class="stat-card">
                    <h4>成功场景</h4>
                    <div class="value">{summary['successful_scenes']}</div>
                </div>
                <div class="stat-card">
                    <h4>平均评分</h4>
                    <div class="value">{summary['average_score']:.1f}</div>
                    <div class="unit">/100</div>
                </div>
                <div class="stat-card">
                    <h4>成功率</h4>
                    <div class="value">{summary['successful_scenes'] / summary['total_scenes'] * 100:.1f}%</div>
                </div>
        """
        html += '</div>'

        # 评分等级判断
        avg_score = summary['average_score']
        if avg_score >= 80:
            grade = "优秀"
            alert_class = "success"
            comment = "系统整体表现优异，各项指标均处于较高水平。"
        elif avg_score >= 60:
            grade = "良好"
            alert_class = "info"
            comment = "系统表现良好，但在某些方面仍有改进空间。"
        elif avg_score >= 40:
            grade = "中等"
            alert_class = "warning"
            comment = "系统整体表现中等，建议重点关注评分较低的领域。"
        else:
            grade = "需改进"
            alert_class = "danger"
            comment = "系统评分较低，建议全面审视和改进。"

        html += f"""
                <div class="alert {alert_class}">
                    <strong>整体等级: {grade}</strong><br>
                    {comment}
                </div>
            </div>
        </div>
        <div class="page-break"></div>
        """

        return html

    def _build_methodology(self) -> str:
        """构建评价方法论"""
        return """
        <div class="content">
            <div class="section">
                <h1>2. 评价方法论</h1>

                <h2>2.1 指标体系概述</h2>
                <p>本评价框架采用了多维度的指标体系，主要包括以下几个类别：</p>

                <h2>2.2 数据质量指标 (Data Quality)</h2>
                <ul>
                    <li><strong>数据完整性 (Data Integrity)</strong>: 评估JSON文件中必需字段的完整性</li>
                    <li><strong>数据一致性 (Data Consistency)</strong>: 检查节点-边关系的逻辑一致性</li>
                    <li><strong>图复杂度 (Graph Complexity)</strong>: 评估图结构的规模和密度</li>
                    <li><strong>节点属性丰富度 (Node Attribute Richness)</strong>: 评估节点信息的完整性</li>
                </ul>

                <h2>2.3 报告质量指标 (Report Quality)</h2>
                <ul>
                    <li><strong>报告完整性 (Report Completion)</strong>: 检查报告是否包含所有必要部分</li>
                    <li><strong>信息密度 (Information Density)</strong>: 评估关键信息词汇的密度</li>
                    <li><strong>可读性 (Readability)</strong>: 基于句子长度评估文本可读性</li>
                    <li><strong>事件检测丰富度 (Event Detection Richness)</strong>: 评估报告中事件类型的多样性</li>
                </ul>

                <h2>2.4 端到端指标 (End-to-End Performance)</h2>
                <ul>
                    <li><strong>数据-报告对齐度 (Data-Report Alignment)</strong>: 评估结构化数据是否在报告中得到体现</li>
                    <li><strong>系统综合评分 (System Comprehensive Score)</strong>: 综合所有指标的加权评分</li>
                </ul>

                <h2>2.5 评分标准</h2>
                <p>所有指标采用百分制评分（0-100），其中：</p>
                <ul>
                    <li>80-100: 优秀，表现出色</li>
                    <li>60-79: 良好，表现满意</li>
                    <li>40-59: 中等，需要改进</li>
                    <li>0-39: 需改进，存在明显问题</li>
                </ul>
            </div>
        </div>
        <div class="page-break"></div>
        """

    def _build_overall_results(self, runner: EvaluationRunner) -> str:
        """构建综合评价结果"""
        summary = runner.summary
        results = runner.results

        scene_names = sorted(results.keys())
        scores = [results[name].overall_score for name in scene_names]

        html = """
        <div class="content">
            <div class="section">
                <h1>3. 综合评价结果</h1>

                <h2>3.1 场景评分总览</h2>
        """

        html += '<table>'
        html += '<thead><tr><th>场景</th><th>综合评分</th><th>等级</th><th>状态</th></tr></thead>'
        html += '<tbody>'

        for scene_name in scene_names:
            result = results[scene_name]
            score = result.overall_score

            if score >= 80:
                grade = "优秀"
            elif score >= 60:
                grade = "良好"
            elif score >= 40:
                grade = "中等"
            else:
                grade = "需改进"

            status = "✓ 成功" if result.status == 'success' else "✗ 失败"

            html += f'<tr><td>{scene_name}</td><td>{score:.2f}</td><td>{grade}</td><td>{status}</td></tr>'

        html += '</tbody></table>'

        # 指标统计
        html += '<h2>3.2 指标统计汇总</h2>'
        html += '<table>'
        html += '<thead><tr><th>指标名称</th><th>平均值</th><th>最小值</th><th>最大值</th><th>标准差</th></tr></thead>'
        html += '<tbody>'

        for metric_name in sorted(summary['metrics_statistics'].keys()):
            stats = summary['metrics_statistics'][metric_name]
            html += f"""
            <tr>
                <td>{metric_name}</td>
                <td>{stats['mean']:.2f}</td>
                <td>{stats['min']:.2f}</td>
                <td>{stats['max']:.2f}</td>
                <td>{np.std([stats['mean'], stats['min'], stats['max']]):.2f}</td>
            </tr>
            """

        html += '</tbody></table>'
        html += """
            </div>
        </div>
        <div class="page-break"></div>
        """

        return html

    def _build_scene_details(self, runner: EvaluationRunner) -> str:
        """构建场景详细评价"""
        results = runner.results

        html = """
        <div class="content">
            <div class="section">
                <h1>4. 场景详细评价</h1>
        """

        for scene_name in sorted(results.keys()):
            result = results[scene_name]

            html += f"""
                <h2>4.{list(results.keys()).index(scene_name) + 1} {scene_name}</h2>
                <p><strong>评价时间:</strong> {result.evaluation_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>综合评分:</strong> {result.overall_score:.2f}/100</p>
                <p><strong>评价状态:</strong> {'✓ 成功' if result.status == 'success' else '✗ 失败'}</p>
            """

            if result.status == 'failed':
                html += f"<p><strong>错误信息:</strong> {result.error_message}</p>"
                continue

            # 各类别评分
            for category, metrics in result.metrics_results.items():
                if not metrics:
                    continue

                category_labels = {
                    'data_quality': '数据质量',
                    'report_quality': '报告质量',
                    'end_to_end': '端到端性能',
                    'comprehensive': '综合评分'
                }

                category_label = category_labels.get(category, category)

                html += f"""
                <h3>{category_label}</h3>
                <div>
                """

                for metric_name, metric_result in metrics.items():
                    html += f"""
                    <div class="metric-item">
                        <div class="name">{metric_name}</div>
                        <div class="value">{metric_result.metric_value} {metric_result.unit}</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {min(metric_result.metric_value, 100)}%">
                                {metric_result.metric_value:.1f}
                            </div>
                        </div>
                        <div class="formula">公式: {metric_result.formula}</div>
                        <div class="description">{metric_result.description}</div>
                    </div>
                    """

                html += "</div>"

        html += """
            </div>
        </div>
        <div class="page-break"></div>
        """

        return html

    def _build_detailed_metrics(self, runner: EvaluationRunner) -> str:
        """构建详细指标分析"""
        summary = runner.summary

        html = """
        <div class="content">
            <div class="section">
                <h1>5. 指标详细分析</h1>
        """

        for metric_name, stats in sorted(summary['metrics_statistics'].items()):
            mean_val = stats['mean']
            min_val = stats['min']
            max_val = stats['max']
            count = stats['count']

            # 判断评价
            if mean_val >= 80:
                assessment = "表现优异，无需改进"
                color = "#28a745"
            elif mean_val >= 60:
                assessment = "表现良好，继续保持"
                color = "#17a2b8"
            elif mean_val >= 40:
                assessment = "表现中等，需要改进"
                color = "#ffc107"
            else:
                assessment = "表现较差，需要重点改进"
                color = "#dc3545"

            html += f"""
            <div class="metric-item" style="border-left-color: {color};">
                <div class="name">{metric_name}</div>
                <div style="margin: 10px 0;">
                    <span style="margin-right: 20px;"><strong>平均值:</strong> {mean_val:.2f}</span>
                    <span style="margin-right: 20px;"><strong>最小值:</strong> {min_val:.2f}</span>
                    <span><strong>最大值:</strong> {max_val:.2f}</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {mean_val}%">{mean_val:.1f}</div>
                </div>
                <div class="description"><strong>评价:</strong> {assessment}</div>
            </div>
            """

        html += """
            </div>
        </div>
        <div class="page-break"></div>
        """

        return html

    def _build_recommendations(self, runner: EvaluationRunner) -> str:
        """构建改进建议"""
        results = runner.results

        html = """
        <div class="content">
            <div class="section">
                <h1>6. 改进建议与推荐</h1>
        """

        recommendation_gen = RecommendationGenerator()

        for scene_name in sorted(results.keys()):
            result = results[scene_name]

            if result.status == 'failed':
                continue

            recommendations = recommendation_gen.generate_recommendations(result)

            html += f"""
                <h2>6.{list(results.keys()).index(scene_name) + 1} {scene_name} 的改进建议</h2>
            """

            for category, suggestions in recommendations.items():
                if not suggestions:
                    continue

                category_labels = {
                    'data_quality': '数据质量方面',
                    'report_quality': '报告质量方面',
                    'end_to_end': '端到端性能方面',
                    'general': '总体建议'
                }

                category_label = category_labels.get(category, category)

                html += f"""
                <h3>{category_label}</h3>
                <ul>
                """

                for suggestion in suggestions:
                    html += f"<li>{suggestion}</li>"

                html += "</ul>"

        html += """
            </div>
        </div>
        <div class="page-break"></div>
        """

        return html

    def _build_conclusion(self, runner: EvaluationRunner) -> str:
        """构建结论与展望"""
        summary = runner.summary
        avg_score = summary['average_score']

        if avg_score >= 80:
            conclusion = "系统整体表现优异，各项评价指标均处于较高水平，表明该系统具有较强的可信性和可靠性。建议继续保持当前的质量标准，同时可探索进一步的优化空间。"
        elif avg_score >= 60:
            conclusion = "系统表现良好，大多数指标达到了预期要求。建议重点关注评分较低的领域，采取针对性的改进措施以进一步提升系统性能。"
        elif avg_score >= 40:
            conclusion = "系统整体表现中等，在多个方面存在改进空间。建议建立改进计划，逐步解决存在的问题，重点关注数据质量和报告生成的一致性。"
        else:
            conclusion = "系统评分较低，需要进行全面的检查和改进。建议从数据采集、处理和报告生成等多个方面进行优化。"

        return f"""
        <div class="content">
            <div class="section">
                <h1>7. 结论与展望</h1>

                <h2>7.1 主要结论</h2>
                <p>{conclusion}</p>

                <h2>7.2 后续工作建议</h2>
                <ul>
                    <li>建立持续监测机制，定期对系统进行评估，跟踪改进效果</li>
                    <li>在实际应用中收集用户反馈，进一步优化系统功能</li>
                    <li>探索更加精细化的评价指标，适应不同场景的需求</li>
                    <li>推动系统的实际应用和部署，进行真实场景的验证</li>
                    <li>与业界交流合作，学习最佳实践，持续改进系统</li>
                </ul>

                <h2>7.3 展望</h2>
                <p>该评价框架为交通场景理解系统提供了全面、客观的评估基础。随着系统的不断完善和优化，
                相信其在交通管理、智能出行等领域将发挥越来越重要的作用。我们期待该系统能够为城市交通
                的智能化和高效化做出积极贡献。</p>
            </div>
        </div>
        """

    def _build_footer(self) -> str:
        """构建页脚"""
        return f"""
        <div class="footer">
            <p>报告生成工具: 交通场景理解系统评价框架 v1.0</p>
            <p>生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>

        </div>
        """


class PDFReportGenerator:
    """PDF报告生成器 - 使用reportlab库"""

    def __init__(self, config: ReportConfig):
        self.config = config

    def _register_chinese_font(self):
        """注册中文字体"""
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        import os

        # 尝试查找系统中文字体
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',      # 黑体
            'C:/Windows/Fonts/msyh.ttf',       # 微软雅黑
            'C:/Windows/Fonts/simsun.ttc',     # 宋体
            '/Library/Fonts/SimHei.ttf',       # macOS
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',  # Linux
        ]

        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break

        if font_path:
            try:
                if font_path.endswith('.ttc'):
                    # TTC字体需要指定字体索引
                    pdfmetrics.registerFont(TTFont('SimHei', font_path, subfontIndex=0))
                else:
                    pdfmetrics.registerFont(TTFont('SimHei', font_path))
                return 'SimHei'
            except Exception as e:
                logger.warning(f"注册中文字体失败: {e}")
                return None
        else:
            logger.warning("未找到中文字体，将使用默认字体")
            return None

    def generate_report(self,
                        runner: EvaluationRunner,
                        output_path: Path) -> Optional[Path]:
        """
        生成PDF报告

        Args:
            runner: 评价运行引擎
            output_path: 输出路径

        Returns:
            Path: 报告文件路径，如果失败返回None
        """
        logger.info("生成PDF报告...")

        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
            from reportlab.lib import colors
            from reportlab.pdfgen import canvas

            output_file = output_path / 'evaluation_report.pdf'

            # 注册中文字体
            chinese_font = self._register_chinese_font()

            # 创建PDF文档
            doc = SimpleDocTemplate(str(output_file), pagesize=A4)
            story = []
            styles = getSampleStyleSheet()

            # 创建中文样式
            if chinese_font:
                chinese_title_style = ParagraphStyle(
                    'ChineseTitle',
                    fontName=chinese_font,
                    fontSize=24,
                    textColor=colors.HexColor('#667eea'),
                    spaceAfter=30,
                    alignment=1  # 居中
                )

                chinese_normal_style = ParagraphStyle(
                    'ChineseNormal',
                    fontName=chinese_font,
                    fontSize=12,
                    leading=18,
                )

                chinese_heading_style = ParagraphStyle(
                    'ChineseHeading',
                    fontName=chinese_font,
                    fontSize=16,
                    textColor=colors.HexColor('#667eea'),
                    spaceAfter=12,
                    spaceBefore=12,
                )
            else:
                chinese_title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=24,
                    textColor=colors.HexColor('#667eea'),
                    spaceAfter=30,
                    alignment=1
                )
                chinese_normal_style = styles['Normal']
                chinese_heading_style = styles['Heading2']

            # 添加标题
            story.append(Paragraph(self.config.title, chinese_title_style))
            story.append(Spacer(1, 0.3 * inch))

            # 添加文档信息
            info_text = f"""
            <b>版本:</b> {self.config.version}<br/>
            <b>作者:</b> {self.config.author}<br/>
            <b>生成时间:</b> {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}<br/>
            """
            story.append(Paragraph(info_text, chinese_normal_style))
            story.append(Spacer(1, 0.2 * inch))

            # 添加摘要信息
            summary = runner.summary
            story.append(Paragraph("【评价摘要】", chinese_heading_style))
            summary_text = f"""
            <b>评价时间:</b> {summary['evaluation_time']}<br/>
            <b>总场景数:</b> {summary['total_scenes']}<br/>
            <b>成功场景:</b> {summary['successful_scenes']}<br/>
            <b>平均评分:</b> {summary['average_score']:.2f}/100<br/>
            """
            story.append(Paragraph(summary_text, chinese_normal_style))
            story.append(PageBreak())

            # 添加评价方法论
            story.append(Paragraph("1. 评价方法论", chinese_heading_style))
            methodology_text = """
            本评价框架采用多维度的指标体系，主要包括：<br/>
            <b>数据质量指标：</b><br/>
            - 数据完整性：评估JSON文件中必需字段的完整性<br/>
            - 数据一致性：检查节点-边关系的逻辑一致性<br/>
            - 图复杂度：评估图结构的规模和密度<br/>
            - 节点属性丰富度：评估节点信息的完整性<br/><br/>
            <b>报告质量指标：</b><br/>
            - 报告完整性：检查报告是否包含所有必要部分<br/>
            - 信息密度：评估关键信息词汇的密度<br/>
            - 可读性：基于句子长度评估文本可读性<br/>
            - 事件检测丰富度：评估报告中事件类型的多样性<br/><br/>
            <b>端到端指标：</b><br/>
            - 数据-报告对齐度：评估结构化数据是否在报告中得到体现<br/>
            - 系统综合评分：综合所有指标的加权评分<br/>
            """
            story.append(Paragraph(methodology_text, chinese_normal_style))
            story.append(PageBreak())

            # 添加场景评价结果
            story.append(Paragraph("2. 场景评价结果", chinese_heading_style))
            results = runner.results
            for scene_name in sorted(results.keys()):
                result = results[scene_name]
                scene_text = f"""
                <b>{scene_name}</b><br/>
                - 评价时间: {result.evaluation_time.strftime('%Y-%m-%d %H:%M:%S')}<br/>
                - 综合评分: {result.overall_score:.2f}/100<br/>
                - 状态: {'成功' if result.status == 'success' else '失败'}<br/>
                """
                story.append(Paragraph(scene_text, chinese_normal_style))
                story.append(Spacer(1, 0.1 * inch))
            story.append(PageBreak())

            # 添加结论
            story.append(Paragraph("3. 结论", chinese_heading_style))
            avg_score = summary['average_score']
            if avg_score >= 80:
                conclusion = "系统整体表现优异，各项评价指标均处于较高水平。"
            elif avg_score >= 60:
                conclusion = "系统表现良好，但在某些方面仍有改进空间。"
            elif avg_score >= 40:
                conclusion = "系统整体表现中等，建议重点关注评分较低的领域。"
            else:
                conclusion = "系统评分较低，建议全面审视和改进。"
            story.append(Paragraph(conclusion, chinese_normal_style))
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph("感谢使用交通场景理解系统评价框架！", chinese_normal_style))

            # 构建PDF
            doc.build(story)

            logger.info(f"PDF报告已生成: {output_file}")
            return output_file

        except ImportError:
            logger.warning("reportlab库未安装，无法生成PDF报告。请使用: pip install reportlab")
            return None
        except Exception as e:
            logger.error(f"PDF报告生成失败: {str(e)}")
            return None


class ReportGenerator:
    """报告生成器主类"""

    def __init__(self, config: ReportConfig = None):
        """
        初始化报告生成器

        Args:
            config: 报告配置
        """
        self.config = config or ReportConfig()
        self.html_generator = HTMLReportGenerator(self.config)
        self.pdf_generator = PDFReportGenerator(self.config)

    def generate_all_reports(self,
                             runner: EvaluationRunner,
                             output_path: Path,
                             formats: List[str] = None) -> Dict[str, Path]:
        """
        生成所有格式的报告

        Args:
            runner: 评价运行引擎
            output_path: 输出路径
            formats: 生成的格式列表 ['html', 'pdf']

        Returns:
            Dict: 生成的报告文件路径字典
        """
        if formats is None:
            formats = ['html', 'pdf']

        generated_files = {}

        if 'html' in formats:
            html_file = self.html_generator.generate_report(runner, output_path)
            if html_file:
                generated_files['html'] = html_file

        if 'pdf' in formats:
            pdf_file = self.pdf_generator.generate_report(runner, output_path)
            if pdf_file:
                generated_files['pdf'] = pdf_file

        return generated_files


if __name__ == "__main__":
    # 示例使用
    config = ReportConfig(
        title="交通场景理解系统评价报告",
        author="智慧交通队",
        include_detailed_metrics=True,
        include_recommendations=True
    )

    eval_config = EvaluationConfig(
        data_path='./data',
        output_path='./output'
    )

    runner = EvaluationRunner(eval_config)
    runner.run()

    # 生成报告
    report_gen = ReportGenerator(config)
    generated = report_gen.generate_all_reports(runner, eval_config.output_path, ['html'])

    for format_type, filepath in generated.items():
        logger.info(f"生成的 {format_type} 报告: {filepath}")
