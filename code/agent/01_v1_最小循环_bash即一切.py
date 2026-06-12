#!/usr/bin/env python3
"""
01_v1_最小循环_bash即一切.py — 里程碑 1：mini 骨架
====================================================
对应《Agent Harness 深度拆解》第九节 路线图·里程碑 1，对标 mini-swe-agent：
  while-loop + 唯一工具 bash + 步数上限 + 卡死检测 + 事件日志

设计决策（来自报告的可迁移启示）：
  P1  loop 本体极简——不到 60 行；外部裁判：MAX_STEPS + 连续重复动作检测
  P3  "bash is all you need"——不定义 JSON 工具，模型输出 ```bash 代码块，
      harness 解析执行（提供商无关，任何能输出文本的模型都能跑）
  P3  ACI 反馈致密——观察结果超长时头尾截断，明确标注截断量
  P8  事件溯源雏形——每个 Action/Observation 追加写入 runs/*.jsonl，
      崩溃后可读日志重放调试

运行（默认 Anthropic 后端）：
  export ANTHROPIC_API_KEY=sk-ant-...
  python 01_v1_最小循环_bash即一切.py "统计当前目录下各类文件的数量"

  # OpenAI 兼容后端（DeepSeek/硅基流动/Kimi 等，沿用本仓库 rag 的约定）：
  export AGENT_BACKEND=openai AGENT_BASE_URL=https://api.deepseek.com \
         AGENT_API_KEY=sk-... AGENT_MODEL=deepseek-chat

⚠️ 沙箱是部署期决策（报告 P5）：本脚本会真实执行模型给出的 bash。
   不放心就在容器里跑：
   docker run -it --rm -v "$PWD":/work -w /work python:3.12-slim bash
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "20"))   # P1: 步数预算
MAX_OBS_CHARS = 4000                                   # P3: 观察裁剪阈值
STUCK_REPEATS = 3                                      # P1: 同一命令连续 N 次判卡死

SYSTEM = """你是一个在 Linux shell 里工作的自治 agent。
每轮回复必须二选一：
1. 继续行动：给出简短思考，然后输出恰好一个 ```bash 代码块（一次只做一小步）；
2. 任务完成：输出一行 `DONE: <结论>`，不要再带代码块。
规则：观察结果会回灌给你；命令要幂等、非交互（禁用 vim/top 等）；失败了就换思路，不要原地重试。"""


# ── LLM 调用（唯一的"智能"入口，其余全是确定性代码）──────────────────
def make_llm():
    backend = os.getenv("AGENT_BACKEND", "anthropic")
    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # 读 ANTHROPIC_API_KEY
        model = os.getenv("AGENT_MODEL", "claude-opus-4-8")

        def chat(messages: list[dict]) -> str:
            resp = client.messages.create(
                model=model, max_tokens=16000, system=SYSTEM, messages=messages)
            if resp.stop_reason == "refusal":  # 4.7+ 必须先查 stop_reason
                return "DONE: 模型拒绝了该请求（safety refusal）"
            return "".join(b.text for b in resp.content if b.type == "text")
    else:  # OpenAI 兼容（DeepSeek / SiliconFlow / Moonshot ...）
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("AGENT_BASE_URL"),
                        api_key=os.getenv("AGENT_API_KEY"))
        model = os.getenv("AGENT_MODEL", "deepseek-chat")

        def chat(messages: list[dict]) -> str:
            resp = client.chat.completions.create(
                model=model, max_tokens=8192,
                messages=[{"role": "system", "content": SYSTEM}] + messages)
            return resp.choices[0].message.content or ""
    return chat


# ── 工具执行：bash（P5：副作用唯一入口，将来换 Docker 只改这里）────────
def run_bash(cmd: str) -> str:
    try:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True,
                           text=True, timeout=60)
        out = (r.stdout + r.stderr).strip() or "(无输出)"
        return f"exit_code={r.returncode}\n{out}"
    except subprocess.TimeoutExpired:
        return "错误：命令超时（60s）。换更快的做法。"


def truncate(obs: str) -> str:  # P3: 反馈致密——别让垃圾淹没上下文
    if len(obs) <= MAX_OBS_CHARS:
        return obs
    half = MAX_OBS_CHARS // 2
    return (obs[:half] + f"\n...[截断 {len(obs) - MAX_OBS_CHARS} 字符，"
            f"输出太长，请用 grep/head 缩小范围]...\n" + obs[-half:])


def parse_bash(reply: str) -> str | None:
    m = re.findall(r"```(?:bash|sh)\n(.*?)```", reply, re.DOTALL)
    return m[-1].strip() if m else None


# ── Agent Loop（P1：全部控制逻辑在此，一屏读完）──────────────────────
def main(task: str) -> None:
    chat = make_llm()
    log_path = Path(__file__).parent / "runs" / f"{int(time.time())}.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    log = lambda kind, data: log_path.open("a").write(
        json.dumps({"t": time.time(), "kind": kind, "data": data},
                   ensure_ascii=False) + "\n")

    messages = [{"role": "user", "content": f"任务：{task}"}]
    log("task", task)
    recent_cmds: list[str] = []

    for step in range(1, MAX_STEPS + 1):
        reply = chat(messages)
        messages.append({"role": "assistant", "content": reply})
        log("action", reply)
        print(f"\n── step {step} ──\n{reply}")

        if reply.lstrip().startswith("DONE:"):
            print("\n✅ 任务结束"); return
        cmd = parse_bash(reply)
        if cmd is None:  # 协议违规也是观察的一种——回灌而不是崩溃
            obs = "格式错误：没有找到 ```bash 代码块。请重新输出。"
        else:
            recent_cmds.append(cmd)  # P1: 卡死检测（OpenHands StuckDetector 的极简版）
            if len(recent_cmds) >= STUCK_REPEATS and \
               len(set(recent_cmds[-STUCK_REPEATS:])) == 1:
                print("\n🛑 卡死：同一命令连续重复，终止"); log("stuck", cmd); return
            obs = truncate(run_bash(cmd))
        log("observation", obs)
        print(f"   ↳ {obs[:200]}{'...' if len(obs) > 200 else ''}")
        messages.append({"role": "user", "content": f"观察结果：\n{obs}"})
        # 注意：错误观察留在上下文里不清洗（Manus："keep errors in context"）

    print(f"\n🛑 达到步数上限 {MAX_STEPS}，强制终止")
    log("budget_exhausted", MAX_STEPS)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"用法: python {Path(__file__).name} \"<任务描述>\"")
    main(" ".join(sys.argv[1:]))
