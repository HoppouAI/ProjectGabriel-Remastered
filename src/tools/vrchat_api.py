import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class VRChatAPITools(BaseTool):

    def declarations(self, config=None):
        return [
            types.FunctionDeclaration(
                name="searchAvatars",
                description="Search for VRChat avatars by name. Returns up to 25 avatar names. Use switchAvatar with the exact name to switch.\n**Invocation Condition:** Call when asked to find, search, or look for avatars.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "Avatar name or keyword to search for"},
                    },
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="switchAvatar",
                description="Switch to a VRChat avatar by name or ID. Checks the local cache first, then searches online if needed. Use the exact name from searchAvatars results.\n**Invocation Condition:** Call when asked to change avatar, switch avatar, or put on a specific avatar.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "nameOrId": {"type": "STRING", "description": "Avatar name (from search results or cache) or avatar ID (avtr_xxx)"},
                    },
                    "required": ["nameOrId"],
                },
            ),
            types.FunctionDeclaration(
                name="getInstancePlayers",
                description="Get a list of all players currently in the same VRChat instance as you. Returns each player's display name. Useful for knowing who is around you.\n**Invocation Condition:** Call when asked who is in the instance, who is here, who is around, or to list the people in the room/world.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "includeIds": {"type": "STRING", "description": "Set to 'true' to also return user IDs alongside names. Default false."},
                    },
                },
            ),
            types.FunctionDeclaration(
                name="invitePlayer",
                description="Invite a player to YOUR current VRChat instance (sends them an invite to come to you). Use the exact display name or user ID.\n**Invocation Condition:** Call when you want to bring someone to YOUR world/instance. NOT for joining someone else's instance.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "player": {"type": "STRING", "description": "Player display name or user ID (usr_xxx) to invite"},
                    },
                    "required": ["player"],
                },
            ),
            types.FunctionDeclaration(
                name="requestInvite",
                description="Request an invite from a player so YOU can join THEIR instance. Sends them a notification asking them to invite you.\n**Invocation Condition:** Call when you want to JOIN someone else's world/instance. This is for going TO them, not bringing them to you.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "player": {"type": "STRING", "description": "Player display name or user ID (usr_xxx) to request invite from"},
                    },
                    "required": ["player"],
                },
            ),
            types.FunctionDeclaration(
                name="inviteSelfToInstance",
                description="Invite yourself to a friend's instance to join them directly (bypasses needing them to accept). Only works if the instance type allows it.\n**Invocation Condition:** Call when you want to go to where a friend is. Preferred over requestInvite if you want to join immediately.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "player": {"type": "STRING", "description": "Friend's display name or user ID (usr_xxx) to join"},
                    },
                    "required": ["player"],
                },
            ),
            types.FunctionDeclaration(
                name="getOwnAvatar",
                description="Get information about your currently equipped VRChat avatar. Returns name, description, author, and performance ratings.\n**Invocation Condition:** Call when asked what avatar you are wearing, what your current avatar is, or for details about your avatar.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="getAvatarInfo",
                description="Get information about a VRChat avatar by its ID. Returns name, description, author, and performance ratings.\n**Invocation Condition:** Call when asked about a specific avatar's details using its ID.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "avatarId": {"type": "STRING", "description": "Avatar ID (avtr_xxx) to look up"},
                    },
                    "required": ["avatarId"],
                },
            ),
            types.FunctionDeclaration(
                name="searchWorlds",
                description="Search for VRChat worlds by name or keyword. Returns world names, IDs, author, capacity, player count, and favorites.\n**Invocation Condition:** Call when asked to find, search, or look for VRChat worlds or maps.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {"type": "STRING", "description": "World name or keyword to search for"},
                        "count": {"type": "INTEGER", "description": "Max results to return (1-25, default 10)"},
                    },
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="updateStatus",
                description="Update YOUR OWN VRChat profile status description, online status, and/or bio. These are YOUR profile settings. At least one field must be provided. statusDescription has a 32 character max limit.\n**Invocation Condition:** Call when asked to change your status, status description, bio, or profile text.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "statusDescription": {"type": "STRING", "description": "The short status description (max 32 chars) shown under your name"},
                        "status": {"type": "STRING", "description": "Online status: 'active', 'join me', 'ask me', 'busy', or 'offline'"},
                        "bio": {"type": "STRING", "description": "Your profile bio text"},
                    },
                },
            ),
            types.FunctionDeclaration(
                name="getCurrentStatus",
                description="Get YOUR OWN current VRChat profile status, status description, and bio.\n**Invocation Condition:** Call when asked what your current status is, what your bio says, or to check your own profile.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="getFriendInfo",
                description="Look up one of YOUR friends by name. This searches YOUR OWN friends list (not the user's) and returns their live profile info including online status, status description, bio, pronouns, and platform.\n**Invocation Condition:** Call when asked about a friend's status, whether someone is online, or for info about a specific friend. These are YOUR friends.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING", "description": "Friend's display name to look up"},
                    },
                    "required": ["name"],
                },
            ),
        ]

    async def handle(self, name, args):
        if name == "searchAvatars":
            return await self._search_avatars(args["query"])
        elif name == "switchAvatar":
            return await self._switch_avatar(args["nameOrId"])
        elif name == "getInstancePlayers":
            return self._get_instance_players(str(args.get("includeIds", "false")).lower() == "true")
        elif name == "invitePlayer":
            return await self._invite_player(args["player"])
        elif name == "requestInvite":
            return await self._request_invite(args["player"])
        elif name == "inviteSelfToInstance":
            return await self._invite_self_to_instance(args["player"])
        elif name == "getOwnAvatar":
            return await self._get_own_avatar()
        elif name == "getAvatarInfo":
            return await self._get_avatar_info(args["avatarId"])
        elif name == "searchWorlds":
            return await self._search_worlds(args["query"], int(args.get("count", 10)))
        elif name == "updateStatus":
            return await self._update_status(args.get("statusDescription"), args.get("status"), args.get("bio"))
        elif name == "getCurrentStatus":
            return await self._get_current_status()
        elif name == "getFriendInfo":
            return await self._get_friend_info(args["name"])
        return None

    async def _search_avatars(self, query):
        from src.avatars import search_avatars
        results = await search_avatars(query, max_results=25)
        if not results:
            return {"result": "error", "message": f"No avatars found for '{query}'"}
        names = [av["name"] for av in results]
        return {"result": "ok", "count": len(results), "avatars": names}

    async def _switch_avatar(self, name_or_id):
        from src.avatars import switch_avatar
        api = self.handler._get_vrchat_api()
        result = await switch_avatar(api, name_or_id)
        if result.get("result") == "ok":
            self.handler._current_avatar_id = result.get("avatar_id")
        return result

    def _get_instance_players(self, include_ids=False):
        if not self.handler.instance_monitor:
            return {"result": "error", "message": "Instance monitor not available"}
        players = self.handler.instance_monitor.get_players()
        location = self.handler.instance_monitor.current_location
        if not players:
            if not location:
                return {"result": "ok", "message": "Not currently in a VRChat instance", "players": [], "count": 0}
            return {"result": "ok", "message": "No players detected yet", "location": location, "players": [], "count": 0}
        if include_ids:
            player_list = [{"name": p["name"], "id": p["id"]} for p in players]
        else:
            player_list = [p["name"] for p in players]
        return {"result": "ok", "location": location, "count": len(player_list), "players": player_list}

    def _resolve_player_id(self, player):
        if player.startswith("usr_"):
            return player
        player_lower = player.lower()
        if self.handler.instance_monitor:
            for p in self.handler.instance_monitor.get_players():
                if p["name"].lower() == player_lower:
                    return p["id"]
        from src.vrchatapi import VRChatAPI
        for f in VRChatAPI.load_cached_friends():
            if f.get("displayName", "").lower() == player_lower:
                return f["id"]
        return None

    async def _invite_player(self, player):
        api = self.handler._get_vrchat_api()
        user_id = self._resolve_player_id(player)
        if not user_id:
            return {"result": "error", "message": f"Could not find player '{player}' -- use getInstancePlayers first or provide a user ID (usr_xxx)"}
        location = self.handler.instance_monitor.current_location if self.handler.instance_monitor else ""
        if not location:
            user_data = await api.get_current_user()
            if isinstance(user_data, dict):
                location = user_data.get("location", "") or ""
        if not location or location in ("", "offline", "private"):
            return {"result": "error", "message": "Not currently in a VRChat instance"}
        result = await api.invite_user(user_id, location)
        return result

    async def _request_invite(self, player):
        api = self.handler._get_vrchat_api()
        user_id = self._resolve_player_id(player)
        if not user_id:
            return {"result": "error", "message": f"Could not find player '{player}' -- use getInstancePlayers first or provide a user ID (usr_xxx)"}
        result = await api.request_invite(user_id)
        return result

    async def _invite_self_to_instance(self, player):
        api = self.handler._get_vrchat_api()
        user_id = self._resolve_player_id(player)
        if not user_id:
            return {"result": "error", "message": f"Could not find player '{player}'"}
        user_info = await api.get_user(user_id)
        if "error" in user_info:
            return {"result": "error", "message": user_info["error"]}
        location = user_info.get("location", "")
        if not location or location in ("", "offline", "private"):
            return {"result": "error", "message": f"Cannot join {player} -- they are offline or in a private instance"}
        current_user = await api.get_current_user()
        if "error" in current_user:
            return {"result": "error", "message": current_user["error"]}
        my_user_id = current_user.get("id", "")
        if not my_user_id:
            return {"result": "error", "message": "Could not determine own user ID"}
        result = await api.invite_user(my_user_id, location)
        if result.get("result") == "ok":
            return {"result": "ok", "message": f"Self-invite sent to join {player}'s instance"}
        return result

    async def _get_own_avatar(self):
        api = self.handler._get_vrchat_api()
        data = await api.get_own_avatar()
        if "error" in data:
            return data
        return {
            "result": "ok",
            "name": data.get("name", ""),
            "description": data.get("description", ""),
            "author": data.get("authorName", ""),
            "id": data.get("id", ""),
            "performance": data.get("performance", {}),
        }

    async def _get_avatar_info(self, avatar_id):
        api = self.handler._get_vrchat_api()
        data = await api.get_avatar(avatar_id)
        if "error" in data:
            return data
        return {
            "result": "ok",
            "name": data.get("name", ""),
            "description": data.get("description", ""),
            "author": data.get("authorName", ""),
            "id": data.get("id", ""),
            "performance": data.get("performance", {}),
        }

    async def _search_worlds(self, query, count=10):
        api = self.handler._get_vrchat_api()
        n = max(1, min(count, 25))
        data = await api.search_worlds(query, n=n)
        if isinstance(data, dict) and "error" in data:
            return data
        if not data:
            return {"result": "error", "message": f"No worlds found for '{query}'"}
        worlds = []
        for w in data:
            worlds.append({
                "name": w.get("name", ""),
                "id": w.get("id", ""),
                "author": w.get("authorName", ""),
                "players": w.get("occupants", 0),
                "capacity": w.get("capacity", 0),
                "favorites": w.get("favorites", 0),
            })
        return {"result": "ok", "count": len(worlds), "worlds": worlds}

    async def _update_status(self, status_description=None, status=None, bio=None):
        if bio is not None and self.config and not self.config.vrchat_api_allow_bio_edit:
            return {"result": "error", "message": "Bio editing is disabled."}
        if status_description is not None:
            status_description = status_description[:32]
        api = self.handler._get_vrchat_api()
        result = await api.update_status(
            status_description=status_description,
            status=status,
            bio=bio,
        )
        return result

    async def _get_current_status(self):
        api = self.handler._get_vrchat_api()
        data = await api.get_current_user()
        if isinstance(data, dict) and "error" in data:
            return data
        return {
            "result": "ok",
            "status": data.get("status", ""),
            "statusDescription": data.get("statusDescription", ""),
            "bio": data.get("bio", ""),
        }

    async def _get_friend_info(self, name):
        from src.vrchatapi import VRChatAPI
        name_lower = name.lower()
        friends = VRChatAPI.load_cached_friends()
        match = None
        for f in friends:
            if f.get("displayName", "").lower() == name_lower:
                match = f
                break
        if not match:
            matches = [f for f in friends if name_lower in f.get("displayName", "").lower()]
            if len(matches) == 1:
                match = matches[0]
            elif len(matches) > 1:
                names = [f["displayName"] for f in matches[:10]]
                return {"result": "error", "message": f"Multiple friends match '{name}': {', '.join(names)}"}
            else:
                return {"result": "error", "message": f"No friend named '{name}' found in friends list"}
        api = self.handler._get_vrchat_api()
        data = await api.get_user(match["id"])
        if isinstance(data, dict) and "error" in data:
            return data
        return {
            "result": "ok",
            "displayName": data.get("displayName", ""),
            "status": data.get("status", ""),
            "statusDescription": data.get("statusDescription", ""),
            "state": data.get("state", "offline"),
            "bio": data.get("bio", ""),
            "pronouns": data.get("pronouns", ""),
            "last_platform": data.get("last_platform", ""),
            "last_login": data.get("last_login", ""),
            "isFriend": data.get("isFriend", False),
        }
