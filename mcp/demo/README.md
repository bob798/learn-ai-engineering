# mcp-demo

基于 [Model Context Protocol (MCP)](https://modelcontextprotocol.io) 的 Python Server 示例，实现了让 Claude 搜索本地 Markdown 笔记的能力。

## MCP 是什么

> MCP 就是给 AI 工具（Claude/Cursor）装插件的标准接口。你写一个 Server，声明"我有哪些工具"，Claude 就能调用它们。

```
你的 MCP Server
    ↕ MCP 协议（stdio）
Claude Desktop / Claude Code
```

核心概念：**Tool**（暴露给 Claude 的函数）/ **Server**（运行 Tool 的进程）/ **Client**（Claude Desktop / Claude Code）

---

## 项目结构

```
mcp-demo/
├── hello-server-mcp.py   # 最简示例：hello 工具，用于理解 MCP 基本结构
├── file-server-mcp.py    # 实战示例：搜索本地 Markdown 笔记
├── test_file_server.py   # 使用官方 mcp SDK 测试 file-server
├── test_local_mcp.py     # 使用 fastmcp Client 测试
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 环境要求

Python **3.10+**（推荐 3.13）

macOS 推荐用 pyenv 管理版本：

```bash
# 安装 pyenv
curl https://pyenv.run | bash

# 将以下内容添加到 ~/.zshrc
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

source ~/.zshrc

# 安装并设置 Python 版本
pyenv install 3.13
pyenv local 3.13
```

也可以直接用 [python.org](https://www.python.org/downloads/) 下载安装包，或 `brew install python@3.13`。

### 2. 创建虚拟环境

```bash
python3.13 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 运行 Server

```bash
# 最简示例
.venv/bin/python3 hello-server-mcp.py

# 笔记搜索（需设置笔记路径）
NOTES_PATH=/your/notes/path .venv/bin/python3 file-server-mcp.py
```

---

## 在 Claude Code 中使用

在项目根目录创建 `.mcp.json`（或修改 `settings.local.json`）：

```json
{
  "mcpServers": {
    "pis-notes": {
      "type": "stdio",
      "command": "/path/to/mcp-demo/.venv/bin/python3",
      "args": ["/path/to/mcp-demo/file-server-mcp.py"],
      "env": {
        "NOTES_PATH": "/path/to/your/notes"
      }
    }
  }
}
```

将路径替换为实际路径，然后在 Claude Code 控制台输入 `/mcp` 确认 Server 已注册。

验证：问 Claude「我关于 RAG 的笔记有哪些？」，能看到 `search_notes` 被调用即成功。

## 在 Claude Desktop 中使用

在 `~/Library/Application Support/Claude/claude_desktop_config.json` 中添加：

```json
{
  "mcpServers": {
    "pis-notes": {
      "command": "/path/to/mcp-demo/.venv/bin/python3",
      "args": ["/path/to/mcp-demo/file-server-mcp.py"],
      "env": {
        "NOTES_PATH": "/path/to/your/notes"
      }
    }
  }
}
```

重启 Claude Desktop 后生效。

---

## 常见问题

| 问题 | 解法 |
|---|---|
| 重启后 MCP 连接失败 | 检查配置文件路径，`command` 必须用绝对路径 |
| Server 跑起来但 Claude 看不到工具 | stdio 模式下不要用 `print`，会污染协议通信 |
| Tool 参数类型报错 | 参数必须加类型注解（`query: str`） |
| 不知道 Server 是否注册成功 | 运行 `mcp dev file-server-mcp.py` 进入调试模式 |

---

## 参考资源

| 资源 | 说明 |
|---|---|
| [modelcontextprotocol.io/docs](https://modelcontextprotocol.io/docs) | 官方文档，看 Quickstart 和 Concepts 两节即可 |
| [python-sdk](https://github.com/modelcontextprotocol/python-sdk) | 官方 Python SDK，examples 目录有完整示例 |
| [official servers](https://github.com/modelcontextprotocol/servers) | 官方示例 Server（filesystem/github/postgres） |
| [FastMCP](https://gofastmcp.com) | 更简洁的 MCP Server 写法 |
| [Simon Willison - MCP](https://simonwillison.net/tags/mcp/) | MCP 生态分析，每篇都有代码 |
