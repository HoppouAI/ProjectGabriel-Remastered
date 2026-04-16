"""Quick test for the GabrielVoiceControl Vencord plugin WebSocket server."""
import asyncio
import json
import uuid

async def main():
    try:
        import websockets
    except ImportError:
        print("Install websockets: pip install websockets")
        return

    uri = "ws://127.0.0.1:9473"
    print(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri, open_timeout=5) as ws:
            print("Connected!\n")

            # Test 1: Ping
            nonce = str(uuid.uuid4())
            cmd = {"op": "ping", "nonce": nonce}
            print(f">> {json.dumps(cmd)}")
            await ws.send(json.dumps(cmd))
            resp = await asyncio.wait_for(ws.recv(), timeout=5)
            print(f"<< {resp}\n")

            # Test 2: Get voice state
            nonce = str(uuid.uuid4())
            cmd = {"op": "get_voice_state", "nonce": nonce}
            print(f">> {json.dumps(cmd)}")
            await ws.send(json.dumps(cmd))
            resp = await asyncio.wait_for(ws.recv(), timeout=5)
            data = json.loads(resp)
            print(f"<< {json.dumps(data, indent=2)}\n")

            if data.get("success") and data.get("data"):
                d = data["data"]
                print(f"Voice connected: {d.get('connected')}")
                if d.get('connected'):
                    print(f"  Channel: {d.get('channel_name')} ({d.get('channel_id')})")
                    print(f"  Users: {len(d.get('users', []))}")
                    for u in d.get("users", []):
                        print(f"    - {u['name']} (mute={u['mute']}, deaf={u['deaf']})")

            print("\nAll tests passed!")

    except ConnectionRefusedError:
        print("Connection refused. Is Discord open with GabrielVoiceControl enabled?")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
