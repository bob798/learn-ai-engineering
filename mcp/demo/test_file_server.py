import asyncio
from mcp import ClientSession, StdioServerParameters, stdio_client

async def main():
    # Connect to the local file-server-mcp.py using stdio transport (official mcp SDK)
    server_params = StdioServerParameters(
        command="python",
        args=["file-server-mcp.py"],
    )
    
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the session
            await session.initialize()
            print("✓ Connected to file-server-mcp.py")
            
            # List available tools
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            print(f"\nAvailable tools: {tool_names}")
            
            # Call the search_notes tool
            #result = await session.call_tool("search_notes", {"query": "test"})
            #print(f"\nSearch result: {result}")

asyncio.run(main())
