import logging
from google.genai import types

logger = logging.getLogger(__name__)


class VRChatStatusTool:
    """Check VRChat AI session status and lobby players."""

    def __init__(self, handler):
        self.handler = handler

    def declarations(self):
        return [
            types.FunctionDeclaration(
                name="getVRChatStatus",
                description=(
                    "Check whether the VRChat AI is currently online in VRChat and "
                    "who is in the lobby/instance with it.\n"
                    "**Invocation Condition:** Call when someone asks if you are on VRChat, "
                    "who is in the VRChat lobby, whether you are online in VRChat, or anything "
                    "about your current VRChat presence or the people around you in VRChat."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        if name != "getVRChatStatus":
            return None

        im = self.handler._instance_monitor
        if not im:
            return {"result": "error", "message": "VRChat instance monitor not available"}

        location = im.current_location
        players = im.get_players()

        if not location:
            return {
                "result": "ok",
                "is_in_world": False,
                "message": "Not currently in a VRChat world",
                "player_count": 0,
                "players": [],
            }

        player_names = [p.get("name", "Unknown") for p in players]
        return {
            "result": "ok",
            "is_in_world": True,
            "location": location,
            "player_count": len(players),
            "players": player_names,
        }
