# Mini Agent Harness · 从 0 到 1 的可运行骨架

> 《[Agent Harness 深度拆解](../../content/02-agent/research/agent-harness-teardown.md)》第九节路线图的代码落地。
> 每个版本对应一个里程碑，每行"复杂度"都能在报告里找到出处。

## 文件

| 文件 | 里程碑 | 对标 | 新增能力 |
|---|---|---|---|
| `01_v1_最小循环_bash即一切.py` | M1 | mini-swe-agent | while-loop · bash 唯一工具 · 步数预算 · 卡死检测 · 观察截断 · JSONL 事件日志 |
| `02_v2_编辑与验证.py` | M2 | Aider | SEARCH/REPLACE 编辑 · 匹配失败回灌 · py_compile/测试验证 · reflection≤3 · git auto-commit |

里程碑 3（事件溯源 + Condenser）与里程碑 4（checkpoint + interrupt + subagent）刻意没写——
报告的结论就是：**只在真有长任务需求时才做**，先用 M1/M2 跑你自己的任务，让痛点告诉你下一步。

## 快速开始

```bash
pip install anthropic python-dotenv        # OpenAI 兼容后端则: pip install openai

# Anthropic（默认，模型 claude-opus-4-8）
export ANTHROPIC_API_KEY=sk-ant-...
python 01_v1_最小循环_bash即一切.py "统计本目录各扩展名文件数量并找出最大的文件"

# 国内 OpenAI 兼容后端（与 code/rag 的提供商约定一致）
export AGENT_BACKEND=openai \
       AGENT_BASE_URL=https://api.deepseek.com \
       AGENT_API_KEY=sk-... \
       AGENT_MODEL=deepseek-chat
python 01_v1_最小循环_bash即一切.py "..."
```

V2（改代码任务，建议在一个有 git 的练习项目里跑）：

```bash
cd ~/some-practice-repo
python /path/to/02_v2_编辑与验证.py "给 utils.py 的 parse_date 增加 ISO8601 支持"
# 可选验证回路与开关：
export AGENT_TEST_CMD="python -m pytest -q"   # 编辑后自动跑测试，失败回灌
export AGENT_GIT=0                            # 关闭自动 commit
```

## 环境变量一览

| 变量 | 默认 | 说明 |
|---|---|---|
| `AGENT_BACKEND` | `anthropic` | `anthropic` 或 `openai`（兼容协议） |
| `AGENT_MODEL` | `claude-opus-4-8` / `deepseek-chat` | 模型 id |
| `AGENT_BASE_URL` / `AGENT_API_KEY` | — | OpenAI 兼容后端专用 |
| `AGENT_MAX_STEPS` | 20 / 25 | P1 步数预算 |
| `AGENT_TEST_CMD` | 无 | V2：编辑后执行的测试命令 |
| `AGENT_GIT` | `1` | V2：自动 commit 开关 |

## 沙箱（P5：隔离级别是部署期决策，不写死在代码里）

```bash
docker run -it --rm -v "$PWD":/work -w /work python:3.12-slim bash
pip install anthropic && export ANTHROPIC_API_KEY=... && python 01_*.py "任务"
```

## 设计决策 ↔ 报告索引

| 代码位置 | 决策 | 报告出处 |
|---|---|---|
| `SYSTEM` 提示 + `parse_bash` | 文本协议而非 JSON tool-calling：提供商无关、协议违规可回灌 | §5 mini-swe-agent |
| `truncate()` | 观察头尾截断 + 标注截断量 + 提示缩小范围 | §5 ACI"反馈致密/去噪" |
| `recent_cmds` 重复检测 | 模型会死循环，harness 必须有外部裁判 | §3 OpenHands StuckDetector |
| 错误观察不清洗、原样回灌 | 模型看见自己的失败才不会重蹈覆辙 | §7 Manus |
| `runs/*.jsonl` | 事件溯源雏形：调试=读日志 | §3 OpenHands 事件流 |
| `EDIT_RE` SEARCH/REPLACE | 定位从坐标空间移到内容空间 | §2 Aider edit format |
| `closest_snippet()` 失败回灌 | 编辑失败是对话的一部分，不是异常 | §2 Aider |
| `verify()` + `MAX_REFLECTIONS=3` | 验证回路决定可靠性上限；3 次修不好就停 | §2 Aider（源码核实） |
| `git_commit()` | git 是免费的 checkpoint 系统 | §2 / 最佳实践 P6 |

## 下一步练习（按报告路线图）

1. **建自有小基准**：写 10-20 个 `(任务, 验收脚本)` 对，跑 V1 记下通过率——这是你所有后续改动的地平线；
2. M3：把 `messages` 数组改造成显式 Action/Observation 事件流 + 加一个可插拔 Condenser（窗口截断 → LLM 摘要）；
3. M4：每步把事件流落盘成 checkpoint，支持 `--resume <run.jsonl>` 断点续跑。
