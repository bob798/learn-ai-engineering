import asyncio
from fastmcp import Client

# 连接本地 Server 脚本
client = Client("file-server-mcp.py")

async def main():
    async with client:
        await client.ping()

        tools = await client.list_tools()
        print(f"Available tools: {[t.name for t in tools]}")

        result = await client.call_tool("search_notes", {"query": "RAG"})
        print(result)

asyncio.run(main())
