import logging
from google.genai import types

logger = logging.getLogger(__name__)


class RelayTool:
    """Relay messages to the main VRChat Gemini Live session."""

    def __init__(self, handler):
        self.handler = handler

    def declarations(self):
        return [
            types.FunctionDeclaration(
                name="relayToVRChat",
                description=(
                    "Send a message to the main VRChat AI session (your other self). "
                    "The VRChat AI is a separate instance of you -- treat it like messaging yourself. "
                    "Be specific and actionable in your content: include WHO is asking, WHAT they want, "
                    "and any relevant details so your VRChat self can act immediately without needing to ask followups. "
                    "Bad: 'play a song'. Good: 'BarricadeBandit wants you to play Blinding Lights by The Weeknd'.\n"
                    "**Invocation Condition:** Call when someone on Discord wants to communicate something "
                    "to the VRChat AI, or when you have important info to share."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "content": {"type": "STRING", "description": "The message or info to relay to the VRChat session"},
                        "from_user": {"type": "STRING", "description": "Discord username of the person (if applicable)"},
                        "priority": {"type": "STRING", "description": "low, normal, or high"},
                    },
                    "required": ["content"],
                },
            ),
        ]

    async def handle(self, name, args):
        if name != "relayToVRChat":
            return None

        content = args.get("content", "")
        from_user = args.get("from_user", "")
        priority = args.get("priority", "normal")

        if not content:
            return {"result": "error", "message": "content required"}

        # Build relay message
        prefix = f"[From Discord, relayed by your Discord self on behalf of {from_user}] " if from_user else "[From your Discord self] "
        relay_text = f"{prefix}{content}"

        if self.handler._relay_callback:
            try:
                await self.handler._relay_callback(relay_text)
                logger.info(f"Relayed to VRChat: {relay_text[:80]}")
                return {"result": "ok", "relayed": True}
            except Exception as e:
                logger.error(f"Relay failed: {e}")
                return {"result": "error", "message": f"Relay failed: {e}"}

        return {"result": "error", "message": "Relay not configured (main session not connected)"}
