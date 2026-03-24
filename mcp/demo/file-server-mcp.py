from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP("pis-notes")
PIS_PATH = os.environ.get("NOTES_PATH", "/Users/bob/workspace/pis")

@mcp.tool()
def search_notes(query: str) -> str:
    """搜索 pis 笔记，返回包含关键词的文件内容片段"""
    results = []
    for root, _, files in os.walk(PIS_PATH):
        for f in files:
            if not f.endswith(".md"):
                continue
            path = os.path.join(root, f)
            content = open(path).read()
            if query.lower() in content.lower():
                snippet = _extract_snippet(content, query)
                results.append(f"## {f}\n{snippet}")
    return "\n\n".join(results) if results else "未找到相关笔记"

def _extract_snippet(content, query, context=200):
    idx = content.lower().find(query.lower())
    start = max(0, idx - context)
    end = min(len(content), idx + context)
    return content[start:end]

if __name__ == "__main__":
    mcp.run()