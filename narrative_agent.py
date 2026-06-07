#!/usr/bin/env python3
"""narrative_agent.py — Stage 3: Trusted natural-language report generation.

Pipeline:
  1. serialize_graph_to_facts() — ST-Graph JSON → structured fact list
  2. LLM inference via Ollama qwen2.5:7b
  3. guardrail_check() — banned-keyword scan with negation awareness (max 2 retries)
  4. interactive_qa_session() — optional RAG-style Q&A against scene data
"""

import sys, json, logging
from typing import Optional
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("NarrativeAgent")

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
MAX_RETRIES = 2
REQUEST_TIMEOUT = 300

SYSTEM_PROMPT = """[最高系统禁令]:
你是严谨的交通案件文书。所有叙事素材必须100%严格基于【绝对已知事实列表】，严禁脑补。
若事实中未提及异常事件，报告中绝不允许编造。

请严格按以下格式输出报告:
## 交通场景报告

### 一、场景概览
- 监控时长: [时长]
- 涉及车辆: [列表]

### 二、车辆行为时序
按时间顺序列出每辆车的运动状态、转向语义、纵向趋势。

### 三、安全评估
基于事实给出安全评估。"""

BANNED_KEYWORDS = [
    # Only ban fabricated dangerous words; collision risks are now fact-based
    "伤亡", "死亡", "重伤", "轻伤", "丧生", "遇难",
    "违法", "违章", "逃逸", "酒驾", "毒驾",
]


class TrafficNarrativeSystem:
    """Stage 3: consumes ST-Graph JSON → trusted natural-language report."""

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE):
        self.model = model
        self.base_url = base_url
        self._current_json: Optional[dict] = None
        logger.info(f"[NARRATIVE] model={model}")

    # ── Module 1: Fact serialiser ────────────────────────────

    @staticmethod
    def serialize_graph_to_facts(json_data: dict) -> str:
        """Convert ST-Graph JSON into a compact fact list for the LLM."""
        from collections import Counter

        meta = json_data.get("meta_info", {})
        nodes = json_data.get("nodes", {})
        vehicles = {k: v for k, v in nodes.items() if v.get("type") == "vehicle"}

        turn_dist = Counter(v.get("turn_semantic", "?") for v in vehicles.values())
        fsm_dist = Counter(v.get("final_state", "?") for v in vehicles.values())
        follow_count = sum(1 for v in vehicles.values() if v.get("following"))

        facts = [
            f"事实：监控时长 {meta.get('total_duration_secs', 0)} 秒，"
            f"{meta.get('total_frames_processed', 0)} 帧，共 {len(vehicles)} 辆车。",
            f"事实：转向 — 直行 {turn_dist.get('Go Straight', 0)}，"
            f"左转 {turn_dist.get('Left Turn', 0)}，右转 {turn_dist.get('Right Turn', 0)}。",
            f"事实：运动 — 加速 {fsm_dist.get('Accelerating', 0)}，"
            f"蠕行 {fsm_dist.get('Creeping', 0)}，巡驶 {fsm_dist.get('Cruising', 0)}，"
            f"减速 {fsm_dist.get('Decelerating', 0)}，静止 {fsm_dist.get('Stationary', 0)}。",
        ]
        if follow_count:
            facts.append(f"事实：{follow_count} 辆车存在跟车关系。")

        turning = [(nid, info) for nid, info in vehicles.items()
                   if info.get("turn_semantic") in ("Left Turn", "Right Turn")]
        if turning:
            facts.append(f"事实：转弯车辆共 {len(turning)} 辆：")
            for node_id, info in turning:
                facts.append(
                    f"  - {node_id}：{info.get('turn_semantic')}，"
                    f"{info.get('first_seen', '?')}→{info.get('last_seen', '?')}，"
                    f"{info.get('final_state', '?')}"
                )

        # ---- Collision risk events from Stage 2 TTC analysis ----
        event_chains = json_data.get("temporal_event_chains", [])
        collision_events = [
            e for e in event_chains
            if e.get("interaction", {}).get("event_type") == "Collision_Risk"
        ]
        if collision_events:
            # Aggregate: keep only top critical/high severity events
            critical = [e for e in collision_events
                       if e.get("interaction", {}).get("severity") == "Critical"]
            high = [e for e in collision_events
                   if e.get("interaction", {}).get("severity") == "High"]
            moderate = [e for e in collision_events
                       if e.get("interaction", {}).get("severity") == "Moderate"]

            facts.append(f"\n事实：检测到碰撞风险事件 {len(collision_events)} 次"
                        f"（严重: {len(critical)}, 高危: {len(high)}, 中等: {len(moderate)}）。")

            # Report all critical events
            for evt in critical[:20]:
                inter = evt.get("interaction", {})
                pred = inter.get("causal_prediction", {})
                facts.append(
                    f"  - [严重碰撞风险] 时间: {evt.get('time_span', '?')}"
                    f" | TTC: {pred.get('min_TTC', '?')}"
                    f" | 车辆: {', '.join(f.get('node', '?') for f in evt.get('facts', []))}"
                )

            # Report top 10 high-risk events
            for evt in high[:10]:
                inter = evt.get("interaction", {})
                pred = inter.get("causal_prediction", {})
                facts.append(
                    f"  - [高危碰撞风险] 时间: {evt.get('time_span', '?')}"
                    f" | TTC: {pred.get('min_TTC', '?')}"
                    f" | 车辆: {', '.join(f.get('node', '?') for f in evt.get('facts', []))}"
                )

            # Summarize moderate risks
            if moderate:
                unique_pairs = set()
                for evt in moderate:
                    nodes = tuple(sorted(f.get("node", "?") for f in evt.get("facts", [])))
                    unique_pairs.add(nodes)
                facts.append(f"  - [中等风险] 涉及 {len(unique_pairs)} 对不同车辆对，共 {len(moderate)} 次。")
        else:
            facts.append("\n事实：未检测到碰撞风险事件。")

        return "\n".join(facts)

    # ── Module 2: LLM inference ──────────────────────────────

    def _call_ollama(self, prompt: str, system_prompt: str = SYSTEM_PROMPT,
                     stream: bool = True, temperature: float = 0.3) -> str:
        full_prompt = f"{system_prompt}\n\n## 绝对已知事实列表\n{prompt}\n\n请基于以上事实生成报告："
        payload = {
            "model": self.model, "prompt": full_prompt, "stream": stream,
            "options": {"temperature": temperature, "num_predict": 4096},
        }
        try:
            if stream:
                return self._stream_response(payload)
            resp = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            return (resp.json().get("response", "") or
                    resp.json().get("thinking", "[系统]: 未生成有效响应"))
        except requests.exceptions.ConnectionError:
            logger.error("[LLM] 无法连接 Ollama")
            return "[ERROR] Ollama 服务不可用"
        except Exception as e:
            logger.error(f"[LLM] 调用失败: {e}")
            return f"[ERROR] {e}"

    def _stream_response(self, payload: dict) -> str:
        """Stream tokens from Ollama, with thinking-token fallback."""
        tokens = []
        all_data = {"response": "", "thinking": ""}
        try:
            with requests.post(
                f"{self.base_url}/api/generate", json=payload,
                stream=True, timeout=REQUEST_TIMEOUT,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("thinking"):
                        all_data["thinking"] += chunk["thinking"]
                    if chunk.get("response"):
                        sys.stdout.write(chunk["response"])
                        sys.stdout.flush()
                        tokens.append(chunk["response"])
                        all_data["response"] += chunk["response"]
                    if chunk.get("done"):
                        break
        except Exception as e:
            logger.error(f"[STREAM] {e}")

        result = "".join(tokens)
        if not result.strip():
            result = all_data.get("response", "") or all_data.get("thinking", "")
        return result

    # ── Module 3: Guardrail ──────────────────────────────────

    def guardrail_check(self, llm_text: str, _json_data: dict) -> tuple:
        """Scan output for hallucinated banned keywords (negation-aware)."""
        negation_prefixes = ["未", "没有", "无", "非", "不", "否", "未发现", "未检测到"]
        triggered = []
        for keyword in BANNED_KEYWORDS:
            idx = 0
            while True:
                idx = llm_text.find(keyword, idx)
                if idx == -1:
                    break
                prefix = llm_text[max(0, idx - 5):idx]
                if not any(neg in prefix for neg in negation_prefixes):
                    triggered.append(keyword)
                    break
                idx += len(keyword)
        if not triggered:
            return True, llm_text
        logger.warning(f"[GUARDRAIL] 敏感词: {triggered}，触发重新生成")
        return False, llm_text

    # ── Module 4: Interactive QA ─────────────────────────────

    def interactive_qa_session(self, user_query: str,
                               json_data: Optional[dict] = None) -> str:
        """Answer questions strictly from scene data (RAG-style)."""
        if json_data is None:
            json_data = self._current_json
        if json_data is None:
            return "[ERROR] 没有已加载的交通场景数据。"

        context = self.serialize_graph_to_facts(json_data)
        qa_system = (
            "你是交通监控数据分析师。严格基于【场景上下文数据】回答问题。"
            "如果数据中找不到答案，如实回答'数据中未记录该信息'。禁止编造。"
        )
        qa_prompt = (
            f"{qa_system}\n\n## 场景上下文数据\n{context}\n\n"
            f"## 当前问题\n用户: {user_query}\n\n助手: 基于数据回答："
        )
        return self._call_ollama(qa_prompt, system_prompt="", stream=True, temperature=0.3)

    # ── Public API ───────────────────────────────────────────

    def generate_official_report(self, st_graph_json: dict,
                                 max_retries: int = MAX_RETRIES) -> str:
        """Full pipeline: serialise → LLM → guardrail → report."""
        self._current_json = st_graph_json
        logger.info("[NARRATIVE] 序列化 JSON → 事实列表...")
        facts_text = self.serialize_graph_to_facts(st_graph_json)

        for attempt in range(max_retries + 1):
            logger.info(f"[NARRATIVE] LLM 推理 第{attempt + 1}次...")
            print("\n" + "=" * 60)
            print("  交通场景分析报告")
            print("=" * 60 + "\n")
            report = self._call_ollama(facts_text, stream=True, temperature=0.3)
            print("\n" + "=" * 60)

            is_safe, report = self.guardrail_check(report, st_graph_json)
            if is_safe:
                logger.info("[NARRATIVE] 护栏检查通过")
                break
            logger.warning(f"[NARRATIVE] 护栏触发，重试 {attempt + 1}/{max_retries}")
            if attempt >= max_retries:
                logger.error("[NARRATIVE] 达到最大重试次数")
                break
        return report


def run_stage3(st_graph_path: str, output_dir: str = "output"):
    """Standalone entry point: load ST-Graph JSON → generate trusted narrative report.

    Args:
        st_graph_path: path to Stage 2 ST-Graph JSON file
        output_dir: directory for output report

    Returns:
        report text string
    """
    print(f"[STAGE3] Loading {st_graph_path}")
    with open(st_graph_path, "r", encoding="utf-8") as f:
        st_graph = json.load(f)

    system = TrafficNarrativeSystem()
    report = system.generate_official_report(st_graph)

    # Save
    import os as _os
    _os.makedirs(output_dir, exist_ok=True)
    stem = _os.path.splitext(_os.path.basename(st_graph_path))[0]
    report_path = _os.path.join(output_dir, f"traffic_report_{stem}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[STAGE3] Saved: {report_path}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3: Narrative Agent")
    parser.add_argument("st_graph", nargs="?", help="Path to ST-Graph JSON file")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--qa", help="Interactive QA: ask a question about loaded scene")
    args = parser.parse_args()

    if args.st_graph:
        report = run_stage3(args.st_graph, args.output)
        if args.qa:
            system = TrafficNarrativeSystem()
            with open(args.st_graph, "r", encoding="utf-8") as f:
                st_graph = json.load(f)
            answer = system.interactive_qa_session(args.qa, st_graph)
            print(f"\n[QA] {answer}")
    else:
        # Demo: find all ST-Graph JSONs in default locations
        import glob as _glob
        jsons = _glob.glob("st_graph_output*.json") + _glob.glob("output/st_graph_*.json")
        if not jsons:
            print("[STAGE3] No ST-Graph JSON files found. Run Stage 2 first, or specify path.")
            print("[STAGE3] Usage: python narrative_agent.py <path/to/st_graph.json>")
        else:
            for j in jsons:
                run_stage3(j, args.output)
