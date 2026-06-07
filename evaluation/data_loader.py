"""
数据加载器模块
负责加载和解析JSON和TXT文件，支持多种格式和数据验证
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataLoadResult:
    """数据加载结果"""
    success: bool
    data: Union[Dict, str, None]
    file_path: str
    file_type: str
    error_message: Optional[str] = None
    loading_time: float = 0.0
    file_size: int = 0


class JSONLoader:
    """JSON文件加载器"""

    @staticmethod
    def load(file_path: str) -> DataLoadResult:
        """
        加载JSON文件

        Args:
            file_path: JSON文件路径

        Returns:
            DataLoadResult: 加载结果
        """
        start_time = datetime.now()

        try:
            # 检查文件存在性
            if not os.path.exists(file_path):
                return DataLoadResult(
                    success=False,
                    data=None,
                    file_path=file_path,
                    file_type='json',
                    error_message=f"文件不存在: {file_path}"
                )

            # 获取文件大小
            file_size = os.path.getsize(file_path)

            # 加载JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            loading_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"成功加载JSON文件: {file_path}, 大小: {file_size} bytes")

            return DataLoadResult(
                success=True,
                data=data,
                file_path=file_path,
                file_type='json',
                loading_time=loading_time,
                file_size=file_size
            )

        except json.JSONDecodeError as e:
            error_msg = f"JSON解析错误: {str(e)}"
            logger.error(f"{error_msg} - {file_path}")
            return DataLoadResult(
                success=False,
                data=None,
                file_path=file_path,
                file_type='json',
                error_message=error_msg
            )

        except Exception as e:
            error_msg = f"加载失败: {str(e)}"
            logger.error(f"{error_msg} - {file_path}")
            return DataLoadResult(
                success=False,
                data=None,
                file_path=file_path,
                file_type='json',
                error_message=error_msg
            )

    @staticmethod
    def validate(data: Dict) -> Tuple[bool, List[str]]:
        """
        验证JSON数据结构

        Args:
            data: JSON数据

        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误列表)
        """
        errors = []

        # 检查是否为字典
        if not isinstance(data, dict):
            errors.append("数据不是字典类型")
            return False, errors

        # 检查图数据结构
        if 'nodes' not in data:
            errors.append("缺少 'nodes' 字段")
        if 'edges' not in data:
            errors.append("缺少 'edges' 字段")

        # 检查nodes类型
        if 'nodes' in data:
            if not isinstance(data['nodes'], list):
                errors.append("'nodes' 字段应为列表")
            elif len(data['nodes']) == 0:
                errors.append("'nodes' 列表为空")

        # 检查edges类型
        if 'edges' in data:
            if not isinstance(data['edges'], list):
                errors.append("'edges' 字段应为列表")

        is_valid = len(errors) == 0
        return is_valid, errors


class TextLoader:
    """文本文件加载器"""

    @staticmethod
    def load(file_path: str) -> DataLoadResult:
        """
        加载TXT文件

        Args:
            file_path: TXT文件路径

        Returns:
            DataLoadResult: 加载结果
        """
        start_time = datetime.now()

        try:
            # 检查文件存在性
            if not os.path.exists(file_path):
                return DataLoadResult(
                    success=False,
                    data=None,
                    file_path=file_path,
                    file_type='txt',
                    error_message=f"文件不存在: {file_path}"
                )

            # 获取文件大小
            file_size = os.path.getsize(file_path)

            # 加载文本文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = f.read()

            loading_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"成功加载TXT文件: {file_path}, 大小: {file_size} bytes")

            return DataLoadResult(
                success=True,
                data=data,
                file_path=file_path,
                file_type='txt',
                loading_time=loading_time,
                file_size=file_size
            )

        except UnicodeDecodeError:
            # 尝试使用其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    data = f.read()

                loading_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"使用GBK编码成功加载文件: {file_path}")

                return DataLoadResult(
                    success=True,
                    data=data,
                    file_path=file_path,
                    file_type='txt',
                    loading_time=loading_time,
                    file_size=file_size
                )
            except Exception as e:
                error_msg = f"编码错误: {str(e)}"
                logger.error(f"{error_msg} - {file_path}")
                return DataLoadResult(
                    success=False,
                    data=None,
                    file_path=file_path,
                    file_type='txt',
                    error_message=error_msg
                )

        except Exception as e:
            error_msg = f"加载失败: {str(e)}"
            logger.error(f"{error_msg} - {file_path}")
            return DataLoadResult(
                success=False,
                data=None,
                file_path=file_path,
                file_type='txt',
                error_message=error_msg
            )

    @staticmethod
    def validate(data: str) -> Tuple[bool, List[str]]:
        """
        验证文本数据

        Args:
            data: 文本数据

        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误列表)
        """
        errors = []

        # 检查是否为字符串
        if not isinstance(data, str):
            errors.append("数据不是字符串类型")
            return False, errors

        # 检查是否为空
        if len(data.strip()) == 0:
            errors.append("文本内容为空")

        # 检查最小长度
        if len(data) < 10:
            errors.append("文本内容过短")

        is_valid = len(errors) == 0
        return is_valid, errors


class DatasetLoader:
    """数据集加载器 - 支持批量加载"""

    def __init__(self, base_path: str):
        """
        初始化数据集加载器

        Args:
            base_path: 数据集基础路径
        """
        self.base_path = Path(base_path)
        self.json_loader = JSONLoader()
        self.text_loader = TextLoader()
        self.loaded_data = {}

    def load_scene(self, scene_name: str) -> Dict[str, DataLoadResult]:
        """
        加载一个场景的所有数据（JSON和TXT）

        Args:
            scene_name: 场景名称 (e.g., 'S06_c041')

        Returns:
            Dict[str, DataLoadResult]: 加载结果字典
        """
        results = {}
        scene_path = self.base_path / scene_name

        if not scene_path.exists():
            logger.error(f"场景路径不存在: {scene_path}")
            return results

        # 查找JSON文件（支持多种命名模式）
        json_files = list(scene_path.glob('*graph*.json')) + list(scene_path.glob('*output*.json')) + list(scene_path.glob('*_dicts_*.json'))
        for json_file in json_files:
            result = self.json_loader.load(str(json_file))
            if result.success:
                is_valid, errors = self.json_loader.validate(result.data)
                if not is_valid:
                    logger.warning(f"JSON验证失败: {errors}")
                    result.error_message = str(errors)
            results[f"{json_file.stem}_json"] = result

        # 查找TXT文件
        txt_files = list(scene_path.glob('*_report_*.txt'))
        for txt_file in txt_files:
            result = self.text_loader.load(str(txt_file))
            if result.success:
                is_valid, errors = self.text_loader.validate(result.data)
                if not is_valid:
                    logger.warning(f"TXT验证失败: {errors}")
                    result.error_message = str(errors)
            results[f"{txt_file.stem}_txt"] = result

        self.loaded_data[scene_name] = results
        return results

    def load_multiple_scenes(self, scene_names: List[str]) -> Dict[str, Dict[str, DataLoadResult]]:
        """
        批量加载多个场景

        Args:
            scene_names: 场景名称列表

        Returns:
            Dict: 所有场景的加载结果
        """
        all_results = {}
        for scene_name in scene_names:
            all_results[scene_name] = self.load_scene(scene_name)
        return all_results

    def load_all_scenes(self) -> Dict[str, Dict[str, DataLoadResult]]:
        """
        自动发现并加载基路径下的所有场景

        Returns:
            Dict: 所有场景的加载结果
        """
        all_results = {}

        # 扫描基路径下的目录
        for item in self.base_path.iterdir():
            if item.is_dir():
                scene_name = item.name
                logger.info(f"发现场景: {scene_name}")
                all_results[scene_name] = self.load_scene(scene_name)

        return all_results

    def get_loaded_data(self, scene_name: str, file_key: str = None):
        """
        获取已加载的数据

        Args:
            scene_name: 场景名称
            file_key: 文件键（可选）

        Returns:
            加载的数据或数据字典
        """
        if scene_name not in self.loaded_data:
            return None

        scene_data = self.loaded_data[scene_name]

        if file_key is None:
            return scene_data

        if file_key in scene_data:
            return scene_data[file_key].data

        return None

    def print_summary(self):
        """打印加载数据摘要"""
        print("\n" + "=" * 80)
        print("数据加载摘要".center(80))
        print("=" * 80 + "\n")

        for scene_name, results in self.loaded_data.items():
            print(f"\n【场景: {scene_name}】")
            print("-" * 80)

            for file_key, result in results.items():
                status = "✓ 成功" if result.success else "✗ 失败"
                print(f"  {status} | {result.file_path}")
                print(
                    f"        类型: {result.file_type}, 大小: {result.file_size} bytes, 耗时: {result.loading_time:.3f}s")

                if result.error_message:
                    print(f"        错误: {result.error_message}")

        print("\n" + "=" * 80)


class SmartDataLoader:
    """智能数据加载器 - 自动选择加载方式"""

    def __init__(self, base_path: str):
        """
        初始化智能加载器

        Args:
            base_path: 基础路径
        """
        self.dataset_loader = DatasetLoader(base_path)
        self.scenes_data = {}

    def load_and_prepare(self, scene_names: List[str] = None) -> Dict:
        """
        加载并准备数据

        Args:
            scene_names: 场景名称列表，如为None则加载所有

        Returns:
            Dict: 整理后的数据
        """
        if scene_names is None:
            results = self.dataset_loader.load_all_scenes()
        else:
            results = self.dataset_loader.load_multiple_scenes(scene_names)

        # 整理数据
        prepared_data = {}
        for scene_name, file_results in results.items():
            scene_data = {
                'scene_name': scene_name,
                'graph_data': None,
                'report_text': None,
                'metadata': {
                    'load_status': {},
                    'file_paths': {}
                }
            }

            for file_key, result in file_results.items():
                scene_data['metadata']['load_status'][file_key] = result.success
                scene_data['metadata']['file_paths'][file_key] = result.file_path

                if result.success:
                    if result.file_type == 'json':
                        scene_data['graph_data'] = result.data
                    elif result.file_type == 'txt':
                        scene_data['report_text'] = result.data

            prepared_data[scene_name] = scene_data

        self.scenes_data = prepared_data
        return prepared_data

    def get_scene_pair(self, scene_name: str) -> Tuple[Dict, str]:
        """
        获取一个场景的配对数据（图数据和报告）

        Args:
            scene_name: 场景名称

        Returns:
            Tuple[Dict, str]: (图数据, 报告文本)
        """
        if scene_name not in self.scenes_data:
            logger.error(f"场景不存在: {scene_name}")
            return None, None

        scene_data = self.scenes_data[scene_name]
        return scene_data['graph_data'], scene_data['report_text']

    def get_all_scene_pairs(self) -> List[Tuple[str, Dict, str]]:
        """
        获取所有场景的配对数据

        Returns:
            List[Tuple[str, Dict, str]]: 场景名称、图数据、报告文本的列表
        """
        pairs = []
        for scene_name, scene_data in self.scenes_data.items():
            pairs.append((
                scene_name,
                scene_data['graph_data'],
                scene_data['report_text']
            ))
        return pairs


if __name__ == "__main__":
    # 测试示例
    import sys

    # 示例1：单个文件加载
    print("示例1：单个JSON文件加载")
    print("-" * 80)

    # 示例2：数据集加载
    print("\n示例2：数据集加载")
    print("-" * 80)

    # 假设数据路径为 './data'
    data_path = './data'

    if os.path.exists(data_path):
        loader = DatasetLoader(data_path)
        results = loader.load_all_scenes()
        loader.print_summary()
    else:
        print(f"数据路径不存在: {data_path}")
        print("请确保数据文件已下载到正确的位置")
