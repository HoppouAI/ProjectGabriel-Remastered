"""Desktop FBT puppet test. Drives the FBT/* avatar params over OSC.

The avatar's Action layer flips all tracking to Animation when FBT/Enable
is on, then each FBT float (-1..1) drives a humanoid muscle blend tree.
IMPORTANT: all params at 0 = T-pose (muscle 0 = mid range), so we always
send NEUTRAL as the baseline and layer motion on top of it.
Sign convention is Unity muscle naming, min to max: -1 = first word,
+1 = second word. So Front-Back means -1 leans/swings FORWARD, Down-Up
means -1 is DOWN, In-Out means -1 is IN.

Usage: python scripts/test_fbt_puppet.py [demo] [upper] [loop]
  demo: wave | sway | squat | headbang | all (default all)
  upper: upper body only mode, legs/locomotion stay with VRChat so you
         can walk around (WASD or Gabriel's nav) while it gestures
Ctrl+C exits cleanly (returns avatar to normal tracking).
"""

import math
import sys
import time

from pythonosc.udp_client import SimpleUDPClient

OSC_IP = "127.0.0.1"
OSC_PORT = 9000
PREFIX = "/avatar/parameters/FBT/"
RATE_HZ = 45

# rough natural standing pose, tweak once we see it in game
NEUTRAL = {
    "SpineFB": 0.0, "SpineLR": 0.0, "SpineTW": 0.0,
    "HeadNod": 0.0, "HeadTilt": 0.0, "HeadTurn": 0.0,
    "LArmUp": -0.75, "LArmFB": 0.0, "LArmTW": 0.0,
    "LElbow": 0.85, "LWristUD": 0.0, "LWristIO": 0.0,
    "RArmUp": -0.75, "RArmFB": 0.0, "RArmTW": 0.0,
    "RElbow": 0.85, "RWristUD": 0.0, "RWristIO": 0.0,
    "LLegFB": 0.58, "LLegIO": 0.0, "LKnee": 0.85, "LFootUD": -0.3,
    "RLegFB": 0.58, "RLegIO": 0.0, "RKnee": 0.85, "RFootUD": -0.3,
    "HipsY": 0.0, "HipsPitch": 0.0, "HipsRoll": 0.0,
}

client = SimpleUDPClient(OSC_IP, OSC_PORT)


def send_pose(overrides=None):
    pose = dict(NEUTRAL)
    if overrides:
        pose.update(overrides)
    for name, val in pose.items():
        client.send_message(PREFIX + name, max(-1.0, min(1.0, val)))


def demo_wave(t):
    # arm raised, elbow bent, forearm swings via upper arm twist
    return {
        "RArmUp": 0.45,
        "RArmFB": -0.15,
        "RElbow": -0.35,
        "RArmTW": 0.6 * math.sin(t * 5.0),
        "RWristIO": 0.35 * math.sin(t * 5.0),
        "HeadTilt": 0.12 * math.sin(t * 2.5),
    }


def demo_sway(t):
    return {
        "SpineLR": 0.35 * math.sin(t * 2.0),
        "SpineTW": 0.25 * math.sin(t * 2.0 + 0.6),
        "HeadTilt": -0.2 * math.sin(t * 2.0),
        "LArmUp": -0.65 + 0.1 * math.sin(t * 2.0),
        "RArmUp": -0.65 - 0.1 * math.sin(t * 2.0),
    }


def demo_squat(t):
    depth = 0.5 + 0.5 * math.sin(t * 1.5)  # 0..1
    # hips drop eased, legs shorten on a cosine so linear drop dips underground
    return {
        "HipsY": -0.85 * depth ** 1.6,
        "LKnee": 0.85 - 1.8 * depth,
        "RKnee": 0.85 - 1.8 * depth,
        "LLegFB": 0.58 - 1.2 * depth,
        "RLegFB": 0.58 - 1.2 * depth,
        "LLegIO": 0.15 * depth,
        "RLegIO": 0.15 * depth,
        "SpineFB": -0.35 * depth,
        "LArmFB": -0.5 * depth,
        "RArmFB": -0.5 * depth,
        "LElbow": 0.85 - 0.5 * depth,
        "RElbow": 0.85 - 0.5 * depth,
    }


def demo_headbang(t):
    return {
        "HeadNod": 0.6 * math.sin(t * 8.0),
        "SpineFB": -0.15 + 0.15 * math.sin(t * 8.0),
    }


DEMOS = {
    "wave": demo_wave,
    "sway": demo_sway,
    "squat": demo_squat,
    "headbang": demo_headbang,
}


def main():
    flags = {a for a in sys.argv[1:] if a in ("loop", "upper")}
    args = [a for a in sys.argv[1:] if a not in flags]
    loop = "loop" in flags
    upper = "upper" in flags
    which = args[0] if args else "all"
    order = list(DEMOS) if which == "all" else [which]

    print(f"enabling puppet mode ({'upper body' if upper else 'full body'}), sending to {OSC_IP}:{OSC_PORT}")
    send_pose()
    client.send_message(PREFIX + "Upper", upper)
    client.send_message(PREFIX + "Enable", True)
    time.sleep(0.5)

    dt = 1.0 / RATE_HZ
    try:
        while True:
            for name in order:
                print(f"demo: {name}")
                fn = DEMOS[name]
                start = time.monotonic()
                while time.monotonic() - start < 8.0:
                    t = time.monotonic() - start
                    send_pose(fn(t))
                    time.sleep(dt)
                # ease back to neutral between demos
                send_pose()
                time.sleep(1.0)
            if not loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("disabling puppet mode")
        send_pose()
        time.sleep(0.3)
        client.send_message(PREFIX + "Enable", False)


if __name__ == "__main__":
    main()
