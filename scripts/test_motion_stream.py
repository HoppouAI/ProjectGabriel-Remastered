# stream frames from the motion server into VRChat's FBT puppet rig.
# run on the gabriel pc: python scripts/test_motion_stream.py [server_ip] [prompt]
# type new prompts at the console to switch motion mid-stream, ctrl+c to quit.

import sys
import json
import asyncio
import threading

from pythonosc.udp_client import SimpleUDPClient

OSC_IP, OSC_PORT = "127.0.0.1", 9000
PREFIX = "/avatar/parameters/FBT/"

PARAMS = [
    "SpineFB", "SpineLR", "SpineTW",
    "HeadNod", "HeadTilt", "HeadTurn",
    "LArmUp", "LArmFB", "LArmTW", "LElbow", "LWristUD", "LWristIO",
    "RArmUp", "RArmFB", "RArmTW", "RElbow", "RWristUD", "RWristIO",
    "LLegFB", "LLegIO", "LKnee", "LFootUD",
    "RLegFB", "RLegIO", "RKnee", "RFootUD",
    "HipsY", "HipsPitch", "HipsRoll",
]


async def main():
    import websockets

    server = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "walk in circles"
    uri = f"ws://{server}:8765"

    osc = SimpleUDPClient(OSC_IP, OSC_PORT)
    osc.send_message(PREFIX + "Enable", True)
    print(f"connecting to {uri}")

    async with websockets.connect(uri, max_size=None) as ws:
        print(f"connected, prompt: {prompt!r}")
        await ws.send(json.dumps({"type": "prompt", "text": prompt}))

        loop = asyncio.get_running_loop()

        def stdin_reader():
            for line in sys.stdin:
                text = line.strip()
                if text:
                    asyncio.run_coroutine_threadsafe(
                        ws.send(json.dumps({"type": "prompt", "text": text})), loop)
                    print(f"prompt -> {text!r}")

        threading.Thread(target=stdin_reader, daemon=True).start()

        n = 0
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") != "frame":
                    continue
                params = msg.get("params")
                if not params:
                    if n == 0:
                        print("server is streaming raw only (no muscle_ranges.json), nothing to send")
                    n += 1
                    continue
                for p in PARAMS:
                    if p in params:
                        osc.send_message(PREFIX + p, float(params[p]))
                n += 1
                if n % 90 == 0:
                    print(f"frame {msg['t']}  HipsY={params.get('HipsY', 0):+.2f}  "
                          f"LLegFB={params.get('LLegFB', 0):+.2f}  RArmUp={params.get('RArmUp', 0):+.2f}")
        finally:
            osc.send_message(PREFIX + "Enable", False)
            print("puppet disabled")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
