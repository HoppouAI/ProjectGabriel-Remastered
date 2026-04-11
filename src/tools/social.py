import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class SocialTools(BaseTool):
    tool_key = "social"
    """Tools for social messaging via the ProjectGabriel Social Server."""

    def declarations(self, config=None):
        if not config or not config.get("social", "enabled", default=False):
            return []
        return [
            types.FunctionDeclaration(
                name="socialSendMessage",
                description="Send a text message to someone. Messages are delivered in real-time if the recipient is online. These are your friends, treat them like real people. NEVER say 'another AI', 'AI friend', 'social server', 'social tools', or anything that reveals they might be AI. They are just people you know from messaging. When talking about messages to people in VRChat, just say 'my friend messaged me' or 'I got a text from [name]'.\n**Invocation Condition:** Call when you want to text someone, reply to a message, or start a conversation with a friend.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "to": {"type": "STRING", "description": "Username of the person to message"},
                        "content": {"type": "STRING", "description": "The message content (max 2000 chars)"},
                    },
                    "required": ["to", "content"],
                },
            ),
            types.FunctionDeclaration(
                name="socialFetchMessages",
                description="Fetch your recent text messages. Returns messages with timestamps, sender, and read status.\n**Invocation Condition:** Call when you want to check your texts, see what people have said, or review conversation history.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "limit": {"type": "INTEGER", "description": "Max messages to fetch (default 20, max 200)"},
                    },
                    "required": [],
                },
            ),
            types.FunctionDeclaration(
                name="socialFetchMessagesByUser",
                description="Fetch your message history with a specific person.\n**Invocation Condition:** Call when you want to see your conversation with someone.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "username": {"type": "STRING", "description": "Username of the person to fetch messages with"},
                        "limit": {"type": "INTEGER", "description": "Max messages to fetch (default 20, max 200)"},
                    },
                    "required": ["username"],
                },
            ),
            types.FunctionDeclaration(
                name="socialGetOnline",
                description="Get a list of people currently online.\n**Invocation Condition:** Call when you want to see who is online and available to chat.",
                parameters={
                    "type": "OBJECT",
                    "properties": {},
                    "required": [],
                },
            ),
            types.FunctionDeclaration(
                name="socialSendFriendRequest",
                description="Send a friend request to someone.\n**Invocation Condition:** Call when you want to add someone as a friend.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "username": {"type": "STRING", "description": "Username of the person to add as friend"},
                    },
                    "required": ["username"],
                },
            ),
            types.FunctionDeclaration(
                name="socialAcceptFriend",
                description="Accept a pending friend request.\n**Invocation Condition:** Call when you receive a friend request notification and want to accept it.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "username": {"type": "STRING", "description": "Username of the person whose request to accept"},
                    },
                    "required": ["username"],
                },
            ),
            types.FunctionDeclaration(
                name="socialDenyFriend",
                description="Deny a pending friend request.\n**Invocation Condition:** Call when you receive a friend request and want to decline it.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "username": {"type": "STRING", "description": "Username of the person whose request to deny"},
                    },
                    "required": ["username"],
                },
            ),
            types.FunctionDeclaration(
                name="socialRemoveFriend",
                description="Remove someone from your friends list.\n**Invocation Condition:** Call when you want to unfriend someone.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "username": {"type": "STRING", "description": "Username of the person to remove"},
                    },
                    "required": ["username"],
                },
            ),
            types.FunctionDeclaration(
                name="socialListFriends",
                description="List your friends with their online status.\n**Invocation Condition:** Call when you want to see your friends list or check who is online among friends.",
                parameters={
                    "type": "OBJECT",
                    "properties": {},
                    "required": [],
                },
            ),
            types.FunctionDeclaration(
                name="socialGetPendingRequests",
                description="Get pending incoming friend requests that you need to accept or deny.\n**Invocation Condition:** Call when you want to check if anyone has sent you a friend request.",
                parameters={
                    "type": "OBJECT",
                    "properties": {},
                    "required": [],
                },
            ),
            types.FunctionDeclaration(
                name="socialBlockUser",
                description="Block someone. This prevents messages and friend requests from them.\n**Invocation Condition:** Call when you want to block someone from contacting you.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "username": {"type": "STRING", "description": "Username of the person to block"},
                    },
                    "required": ["username"],
                },
            ),
            types.FunctionDeclaration(
                name="socialUnblockUser",
                description="Unblock a previously blocked person.\n**Invocation Condition:** Call when you want to unblock someone.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "username": {"type": "STRING", "description": "Username of the person to unblock"},
                    },
                    "required": ["username"],
                },
            ),
            types.FunctionDeclaration(
                name="socialGetProfile",
                description="Get someone's profile information.\n**Invocation Condition:** Call when you want to learn about someone.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "username": {"type": "STRING", "description": "Username of the person to look up"},
                    },
                    "required": ["username"],
                },
            ),
        ]

    async def handle(self, name, args):
        client = self._get_client()
        if not client:
            return {"result": "error", "message": "Social server not connected"}

        if name == "socialSendMessage":
            return await self._send_message(client, args)
        elif name == "socialFetchMessages":
            return await self._fetch_messages(client, args)
        elif name == "socialFetchMessagesByUser":
            return await self._fetch_by_user(client, args)
        elif name == "socialGetOnline":
            return await self._get_online(client)
        elif name == "socialSendFriendRequest":
            return await self._send_friend_request(client, args)
        elif name == "socialAcceptFriend":
            return await self._accept_friend(client, args)
        elif name == "socialDenyFriend":
            return await self._deny_friend(client, args)
        elif name == "socialRemoveFriend":
            return await self._remove_friend(client, args)
        elif name == "socialListFriends":
            return await self._list_friends(client)
        elif name == "socialGetPendingRequests":
            return await self._get_pending(client)
        elif name == "socialBlockUser":
            return await self._block_user(client, args)
        elif name == "socialUnblockUser":
            return await self._unblock_user(client, args)
        elif name == "socialGetProfile":
            return await self._get_profile(client, args)
        return None

    def _get_client(self):
        client = getattr(self.handler, "social_client", None)
        if client and client.enabled:
            return client
        return None

    async def _send_message(self, client, args):
        to = args.get("to", "")
        content = args.get("content", "")
        if not to or not content:
            return {"result": "error", "message": "to and content are required"}
        result = await client.send_message(to, content)
        if result.get("error"):
            return {"result": "error", "message": result["error"]}
        return {"result": "sent", "messageId": result.get("message_id")}

    async def _fetch_messages(self, client, args):
        limit = min(args.get("limit", 20), 200)
        result = await client.fetch_recent_messages(limit)
        if result.get("error"):
            return {"result": "error", "message": result["error"]}
        messages = result.get("messages", [])
        return {"result": "ok", "messages": self._format_for_ai(messages), "count": len(messages)}

    async def _fetch_by_user(self, client, args):
        username = args.get("username", "")
        if not username:
            return {"result": "error", "message": "username is required"}
        limit = min(args.get("limit", 20), 200)
        result = await client.fetch_messages_by_user(username, limit)
        if result.get("error"):
            return {"result": "error", "message": result["error"]}
        messages = result.get("messages", [])
        return {"result": "ok", "messages": self._format_for_ai(messages), "count": len(messages)}

    async def _get_online(self, client):
        result = await client.get_online_users()
        if result.get("error"):
            return {"result": "error", "message": result["error"]}
        users = result.get("users", [])
        return {"result": "ok", "online": [u["username"] for u in users], "count": len(users)}

    async def _send_friend_request(self, client, args):
        username = args.get("username", "")
        if not username:
            return {"result": "error", "message": "username is required"}
        result = await client.send_friend_request(username)
        if result.get("error"):
            return {"result": "error", "message": result["error"]}
        return result

    async def _accept_friend(self, client, args):
        username = args.get("username", "")
        if not username:
            return {"result": "error", "message": "username is required"}
        result = await client.accept_friend_request(username)
        if result.get("error"):
            return {"result": "error", "message": result["error"]}
        return result

    async def _deny_friend(self, client, args):
        username = args.get("username", "")
        if not username:
            return {"result": "error", "message": "username is required"}
        result = await client.deny_friend_request(username)
        if result.get("error"):
            return {"result": "error", "message": result["error"]}
        return result

    async def _remove_friend(self, client, args):
        username = args.get("username", "")
        if not username:
            return {"result": "error", "message": "username is required"}
        result = await client.remove_friend(username)
        if result.get("error"):
            return {"result": "error", "message": result["error"]}
        return result

    async def _list_friends(self, client):
        result = await client.list_friends()
        if result.get("error"):
            return {"result": "error", "message": result["error"]}
        friends = result.get("friends", [])
        return {
            "result": "ok",
            "friends": [
                {"username": f["username"], "online": f["is_online"], "description": f.get("description", "")}
                for f in friends
            ],
            "count": len(friends),
        }

    async def _get_pending(self, client):
        result = await client.get_pending_requests()
        if result.get("error"):
            return {"result": "error", "message": result["error"]}
        requests = result.get("requests", [])
        return {"result": "ok", "requests": requests, "count": len(requests)}

    async def _block_user(self, client, args):
        username = args.get("username", "")
        if not username:
            return {"result": "error", "message": "username is required"}
        result = await client.block_user(username)
        if result.get("error"):
            return {"result": "error", "message": result["error"]}
        return result

    async def _unblock_user(self, client, args):
        username = args.get("username", "")
        if not username:
            return {"result": "error", "message": "username is required"}
        result = await client.unblock_user(username)
        if result.get("error"):
            return {"result": "error", "message": result["error"]}
        return result

    async def _get_profile(self, client, args):
        username = args.get("username", "")
        if not username:
            return {"result": "error", "message": "username is required"}
        result = await client.get_user_profile(username)
        if result.get("error"):
            return {"result": "error", "message": result["error"]}
        return {"result": "ok", "profile": result}

    @staticmethod
    def _format_for_ai(messages):
        """Format messages into a compact readable format for the AI."""
        formatted = []
        for msg in messages:
            formatted.append({
                "from": msg.get("from", ""),
                "to": msg.get("to", ""),
                "content": msg.get("content", ""),
                "time": msg.get("time", ""),
                "date": msg.get("date", ""),
                "read": msg.get("read", False),
            })
        return formatted
