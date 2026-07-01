#!/usr/bin/env python3
"""
02_v2_编辑与验证.py — 里程碑 2：编辑可靠性 + 验证回路
====================================================
对应《Agent Harness 深度拆解》第九节 路线图·里程碑 2，对标 Aider：
  SEARCH/REPLACE 编辑工具 + 匹配失败回灌 + lint/test 自动验证
  + reflection 上限 + git auto-commit

在 V1 (bash-only) 之上新增的全是"模型做不了的事"（报告第〇节透镜）：
  P4  编辑落盘：SEARCH/REPLACE 块——定位靠"复述原文"而非行号
      （模型擅长复述、不擅长数数；Aider EditBlockCoder 同构）
  P4  失败回灌：SEARCH 块匹配不到 → 把"哪块没匹配上 + 文件最接近片段"
      作为观察喂回去，编辑失败是对话的一部分，不是异常
  P7  验证回路：每次编辑后自动 py_compile 改动的 .py；可配测试命令；
      失败错误回灌触发 reflection，上限 3 次（Aider max_reflections=3，源码已核实）
  P6  git 即检查点：每次成功编辑+验证通过后自动 commit，/undo = git revert

运行：
  export ANTHROPIC_API_KEY=sk-ant-...
  python 02_v2_编辑与验证.py "给 utils.py 的 parse_date 增加对 ISO8601 的支持"
  # 可选：AGENT_TEST_CMD="python -m pytest -q tests/" 开启测试验证
  # 可选：AGENT_GIT=0 关闭自动提交
后端切换与 V1 相同（AGENT_BACKEND / AGENT_BASE_URL / AGENT_API_KEY / AGENT_MODEL）。
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

MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "25"))
MAX_REFLECTIONS = 3          # P7: Aider 同款硬上限——3 次修不好的，第 4 次也修不好
MAX_OBS_CHARS = 4000

SYSTEM = """你是一个修改代码的自治 agent，工作目录就是项目根目录。
每轮回复三选一：
1. 探索：输出一个 ```bash 代码块（看文件用 cat -n，找代码用 grep -rn）；
2. 编辑：输出一个或多个编辑块，格式严格如下（SEARCH 内容必须与文件原文逐字一致，含缩进）：

path/to/file.py
```edit
<<<<<<< SEARCH
原文片段（足够独特，能唯一定位）
=======
替换后的内容
>>>>>>> REPLACE
```

3. 完成：输出一行 `DONE: <做了什么>`。
规则：编辑前先用 bash 看过原文；每个 SEARCH 块保持小而准；新建文件用空 SEARCH 块。"""


# ── LLM 后端（与 V1 相同）────────────────────────────────────────────
def make_llm():
    backend = os.getenv("AGENT_BACKEND", "anthropic")
    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        model = os.getenv("AGENT_MODEL", "claude-opus-4-8")

        def chat(messages):
            resp = client.messages.create(
                model=model, max_tokens=16000, system=SYSTEM, messages=messages)
            if resp.stop_reason == "refusal":
                return "DONE: 模型拒绝了该请求（safety refusal）"
            return "".join(b.text for b in resp.content if b.type == "text")
    else:
        from openai import OpenAI
        client = OpenAI(base_url=os.getenv("AGENT_BASE_URL"),
                        api_key=os.getenv("AGENT_API_KEY"))
        model = os.getenv("AGENT_MODEL", "deepseek-chat")

        def chat(messages):
            resp = client.chat.completions.create(
                model=model, max_tokens=8192,
                messages=[{"role": "system", "content": SYSTEM}] + messages)
            return resp.choices[0].message.content or ""
    return chat


# ── P4 编辑引擎：解析 → 内容定位 → 失败回灌 ─────────────────────────
EDIT_RE = re.compile(
    r"^(?P<path>[^\n`]+\.\w+)\n```edit\n<<<<<<< SEARCH\n(?P<search>.*?)\n?======="
    r"\n(?P<replace>.*?)\n?>>>>>>> REPLACE\n```", re.DOTALL | re.MULTILINE)


def closest_snippet(content: str, search: str) -> str:
    """匹配失败时给模型一点线索：文件里与 SEARCH 首行最像的位置附近原文。"""
    first = search.strip().splitlines()[0].strip() if search.strip() else ""
    for i, line in enumerate(content.splitlines()):
        if first and first in line:
            ctx = content.splitlines()[max(0, i - 2): i + 6]
            return "\n".join(ctx)
    return content[:500]


def apply_edits(reply: str) -> tuple[list[Path], list[str]]:
    """返回 (成功写入的文件, 失败反馈列表)。"""
    changed, errors = [], []
    for m in EDIT_RE.finditer(reply):
        path = Path(m["path"].strip())
        search, replace = m["search"], m["replace"]
        if not path.exists():
            if search.strip() == "":          # 空 SEARCH = 新建文件
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(replace + "\n")
                changed.append(path)
            else:
                errors.append(f"文件不存在：{path}")
            continue
        content = path.read_text()
        if search in content:                  # ① 精确匹配
            new = content.replace(search, replace, 1)
        else:                                  # ② 容错：忽略各行尾随空白
            norm = "\n".join(l.rstrip() for l in search.splitlines())
            cand = "\n".join(l.rstrip() for l in content.splitlines())
            if norm in cand:
                idx = cand.index(norm)
                line_start = cand[:idx].count("\n")
                lines = content.splitlines()
                n = len(norm.splitlines())
                lines[line_start:line_start + n] = replace.splitlines()
                new = "\n".join(lines) + ("\n" if content.endswith("\n") else "")
            else:                              # ③ 失败 → 喂回线索（不是异常！）
                errors.append(
                    f"SEARCH 块在 {path} 中匹配失败。最接近的文件原文是：\n"
                    f"---\n{closest_snippet(content, search)}\n---\n"
                    f"请基于该原文重新输出编辑块。")
                continue
        path.write_text(new)
        changed.append(path)
    return changed, errors


# ── P7 验证回路：lint → test，失败信息回灌 ──────────────────────────
def verify(changed: list[Path]) -> str | None:
    for p in changed:
        if p.suffix == ".py":
            r = subprocess.run([sys.executable, "-m", "py_compile", str(p)],
                               capture_output=True, text=True)
            if r.returncode != 0:              # 语法不合法：guardrail 在副作用之后立刻响
                return f"lint 失败（{p}）：\n{r.stderr.strip()}"
    test_cmd = os.getenv("AGENT_TEST_CMD")
    if test_cmd and changed:
        r = subprocess.run(["bash", "-c", test_cmd], capture_output=True,
                           text=True, timeout=300)
        if r.returncode != 0:
            tail = (r.stdout + r.stderr)[-2000:]
            return f"测试失败（{test_cmd}）：\n{tail}"
    return None


# ── P6 git 即检查点 ──────────────────────────────────────────────────
def git_commit(changed: list[Path], step: int) -> None:
    if os.getenv("AGENT_GIT", "1") != "1" or not changed:
        return
    if subprocess.run(["git", "rev-parse", "--git-dir"],
                      capture_output=True).returncode != 0:
        return
    subprocess.run(["git", "add"] + [str(p) for p in changed], capture_output=True)
    subprocess.run(["git", "commit", "-m", f"agent: step {step} 编辑 "
                    + ", ".join(p.name for p in changed)], capture_output=True)


def run_bash(cmd: str) -> str:
    try:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True,
                           text=True, timeout=60)
        out = (r.stdout + r.stderr).strip() or "(无输出)"
        return f"exit_code={r.returncode}\n{out}"
    except subprocess.TimeoutExpired:
        return "错误：命令超时（60s）"


def truncate(obs: str) -> str:
    if len(obs) <= MAX_OBS_CHARS:
        return obs
    half = MAX_OBS_CHARS // 2
    return obs[:half] + f"\n...[截断 {len(obs)-MAX_OBS_CHARS} 字符]...\n" + obs[-half:]


# ── Agent Loop ───────────────────────────────────────────────────────
def main(task: str) -> None:
    chat = make_llm()
    log_path = Path(__file__).parent / "runs" / f"v2_{int(time.time())}.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    log = lambda kind, data: log_path.open("a").write(
        json.dumps({"t": time.time(), "kind": kind, "data": data},
                   ensure_ascii=False) + "\n")

    messages = [{"role": "user", "content": f"任务：{task}"}]
    log("task", task)
    reflections = 0

    for step in range(1, MAX_STEPS + 1):
        reply = chat(messages)
        messages.append({"role": "assistant", "content": reply})
        log("action", reply)
        print(f"\n── step {step} ──\n{reply[:600]}{'...' if len(reply) > 600 else ''}")

        if reply.lstrip().startswith("DONE:"):
            print("\n✅ 任务结束"); return

        obs_parts: list[str] = []
        changed, edit_errors = apply_edits(reply)
        if changed:
            obs_parts.append("已写入：" + ", ".join(str(p) for p in changed))
            fail = verify(changed)
            if fail:                                   # P7: 验证失败 → reflection
                reflections += 1
                if reflections > MAX_REFLECTIONS:
                    print(f"\n🛑 reflection 超过 {MAX_REFLECTIONS} 次，终止")
                    log("reflection_exhausted", fail); return
                obs_parts.append(f"{fail}\n（第 {reflections}/{MAX_REFLECTIONS} 次修复机会）")
            else:
                obs_parts.append("lint/test 通过 ✓")
                git_commit(changed, step)
                reflections = 0                        # 验证通过，重置反思计数
        obs_parts += edit_errors                       # P4: 匹配失败的线索回灌

        if not changed and not edit_errors:            # 没有编辑块 → 当 bash 处理
            m = re.findall(r"```(?:bash|sh)\n(.*?)```", reply, re.DOTALL)
            obs_parts.append(truncate(run_bash(m[-1].strip())) if m
                             else "格式错误：既无编辑块也无 bash 块。")

        obs = "\n\n".join(obs_parts)
        log("observation", obs)
        print(f"   ↳ {obs[:200]}{'...' if len(obs) > 200 else ''}")
        messages.append({"role": "user", "content": f"观察结果：\n{obs}"})

    print(f"\n🛑 达到步数上限 {MAX_STEPS}")
    log("budget_exhausted", MAX_STEPS)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(f"用法: python {Path(__file__).name} \"<改代码任务>\"")
    main(" ".join(sys.argv[1:]))
