# one-off: sweep the hips pitch/roll params over osc so we can see which way
# the ingame rig actually leans. watch the avatar and note directions.
# usage: python scripts/test_hips_axes.py

import time
from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 9000)

NEUTRAL = {
    'LArmUp': -0.6, 'LArmFB': 0.1, 'LArmTW': 0.4, 'LElbow': 1.0,
    'RArmUp': -0.6, 'RArmFB': 0.1, 'RArmTW': 0.4, 'RElbow': 1.0,
    'LLegFB': 0.58, 'LKnee': 0.85, 'LFootUD': -0.3,
    'RLegFB': 0.58, 'RKnee': 0.85, 'RFootUD': -0.3,
    'SpineFB': 0.0, 'SpineLR': 0.0, 'SpineTW': 0.0,
    'HeadNod': 0.0, 'HeadTilt': 0.0, 'HeadTurn': 0.0,
    'LWristUD': 0.0, 'LWristIO': 0.0, 'RWristUD': 0.0, 'RWristIO': 0.0,
    'LLegIO': 0.0, 'RLegIO': 0.0,
    'HipsPitch': 0.0, 'HipsRoll': 0.0, 'HipsY': 0.0,
}


def send(name, value):
    client.send_message(f"/avatar/parameters/FBT/{name}", float(value))


def ramp(name, target, seconds=1.5, steps=45):
    for i in range(steps + 1):
        send(name, target * i / steps)
        time.sleep(seconds / steps)


print("enabling puppet, neutral stand...")
for k, v in NEUTRAL.items():
    send(k, v)
client.send_message("/avatar/parameters/FBT/Enable", True)
time.sleep(3)

for name, target, label in (
        ("HipsPitch", 0.5, "PITCH +0.5 (should lean FORWARD)"),
        ("HipsPitch", -0.5, "PITCH -0.5 (should lean BACKWARD)"),
        ("HipsRoll", 0.5, "ROLL +0.5 (should lean to HIS RIGHT)"),
        ("HipsRoll", -0.5, "ROLL -0.5 (should lean to HIS LEFT)"),
):
    print(f"--> {label}")
    ramp(name, target)
    time.sleep(3)
    ramp(name, 0, seconds=0.8)
    time.sleep(1)

print("disabling puppet")
client.send_message("/avatar/parameters/FBT/Enable", False)
