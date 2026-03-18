"""Test script for VRChat instance monitor.
Run this, then join/leave VRChat instances to see live player tracking.
Press Ctrl+C to stop.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.cli import setup_logging, C
setup_logging()

from src.instance_monitor import InstanceMonitor

monitor = InstanceMonitor()
last_count = -1
last_players = set()


async def main():
    global last_count, last_players
    monitor.start()
    print(f"\n  {C.B_CYAN}Instance Monitor Test{C.RST}")
    print(f"  {C.DIM}{'─' * 40}{C.RST}")
    print(f"  Watching VRChat logs for player activity...")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        while True:
            players = monitor.get_players()
            current_names = {p["name"] for p in players}
            location = monitor.current_location
            count = len(players)

            if count != last_count or current_names != last_players:
                if location:
                    print(f"  {C.B_CYAN}Instance:{C.RST} {location}")
                else:
                    print(f"  {C.DIM}Not in an instance{C.RST}")

                # Show joins
                joined = current_names - last_players
                for name in joined:
                    print(f"  {C.B_GREEN}+ {name}{C.RST}")

                # Show leaves
                left = last_players - current_names
                for name in left:
                    print(f"  {C.B_RED}- {name}{C.RST}")

                # Show full player list
                print(f"  {C.DIM}Players ({count}):{C.RST}")
                for p in sorted(players, key=lambda x: x["name"]):
                    print(f"    {C.B_WHITE}{p['name']}{C.RST} {C.DIM}({p['id']}){C.RST}")
                print()

                last_count = count
                last_players = current_names

            await asyncio.sleep(2)
    except KeyboardInterrupt:
        print(f"\n  {C.DIM}Stopped.{C.RST}")
        monitor.stop()


if __name__ == "__main__":
    asyncio.run(main())
