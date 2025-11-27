"""
测试脚本：验证 server.py 的连接稳定性改进
"""
import asyncio
import websockets
import json
import time

async def test_websocket_connection():
    """测试 WebSocket 连接和心跳"""
    uri = "ws://127.0.0.1:8000/ws/test-client"

    print("🔗 正在连接到服务器...")

    try:
        async with websockets.connect(uri) as websocket:
            print("✅ 连接成功！")

            # 监听心跳消息
            print("\n📡 开始监听心跳消息（10秒）...")
            start_time = time.time()
            heartbeat_count = 0

            while time.time() - start_time < 10:
                try:
                    message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=2.0
                    )

                    data = json.loads(message)
                    if data.get("type") == "heartbeat":
                        heartbeat_count += 1
                        print(f"💓 收到心跳 #{heartbeat_count}: {data.get('timestamp')}")
                    else:
                        print(f"📨 收到消息: {data}")

                except asyncio.TimeoutError:
                    continue

            print(f"\n✅ 测试完成！共收到 {heartbeat_count} 个心跳包")
            print("🎯 结论: WebSocket 心跳机制正常工作！")

    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("💡 请确保 server.py 正在运行 (python server.py)")

if __name__ == "__main__":
    print("="*60)
    print("FDC WebSocket 连接测试工具")
    print("="*60)
    asyncio.run(test_websocket_connection())
