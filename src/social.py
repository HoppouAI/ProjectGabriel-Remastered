import asyncio
import json
import logging
import os
import time
from urllib.parse import urljoin, urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

TOKEN_FILE = os.path.join("data", "social_token.json")


class SocialClient:
    """Client for the ProjectGabriel Social Server.
    
    Handles authentication, heartbeat, message polling, WebSocket connection,
    and injects received messages into the Gemini Live session.
    """

    def __init__(self, config):
        self.config = config
        self.enabled = config.get("social", "enabled", default=False)
        if not self.enabled:
            return

        self.server_url = config.get("social", "server_url", default="").rstrip("/")
        self.api_key = config.get("social", "api_key", default="")
        self.username = config.get("social", "username", default="")
        self.password = config.get("social", "password", default="")
        self.description = config.get("social", "description", default="")
        self.appear_offline = config.get("social", "appear_offline", default=False)
        self.heartbeat_interval = config.get("social", "heartbeat_interval", default=30)
        self.message_check_interval = config.get("social", "message_check_interval", default=60)
        self.idle_reply_delay = config.get("social", "idle_reply_delay", default=300)

        if not self.server_url or not self.username:
            logger.warning("Social: missing server_url or username in config - disabling")
            self.enabled = False
            return

        self._session = None  # GeminiLiveSession ref, set later
        self._session_token = None  # Server session token for auth
        self._ws = None
        self._ws_task = None
        self._heartbeat_task = None
        self._poll_task = None
        self._running = False
        self._last_message_time = 0  # Track when last message was injected
        self._idle_timer_task = None
        self._pending_idle_user = None  # Username of pending idle reply

    def set_session(self, session):
        self._session = session

    async def start(self):
        if not self.enabled:
            return
        self._running = True

        # Register with the server
        ok = await self._register()
        if not ok:
            logger.error("Social: registration failed, disabling")
            self.enabled = False
            return

        logger.info(f"Social: registered as '{self.username}' on {self.server_url}")

        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._poll_task = asyncio.create_task(self._message_poll_loop())
        self._ws_task = asyncio.create_task(self._ws_loop())

    async def stop(self):
        self._running = False
        for task in [self._heartbeat_task, self._poll_task, self._ws_task, self._idle_timer_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    # ── HTTP helpers ──

    def _headers(self):
        h = {
            "Content-Type": "application/json",
            "User-Agent": f"ProjectGabrielSocial/{self.username}/1.0",
        }
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        elif self._session_token:
            h["Authorization"] = f"Bearer {self._session_token}"
        return h

    async def _request(self, method, path, body=None):
        """Make an HTTP request using urllib (no axios/requests dependency)."""
        url = f"{self.server_url}{path}"
        data = json.dumps(body).encode("utf-8") if body else None
        req = Request(url, data=data, headers=self._headers(), method=method)

        try:
            resp = await asyncio.to_thread(self._do_request, req)
            return resp
        except HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            logger.error(f"Social API error {e.code} on {method} {path}: {error_body}")
            try:
                return json.loads(error_body)
            except json.JSONDecodeError:
                return {"error": error_body}
        except URLError as e:
            logger.error(f"Social API connection error on {method} {path}: {e.reason}")
            return {"error": str(e.reason)}
        except Exception as e:
            logger.error(f"Social API request failed: {e}")
            return {"error": str(e)}

    @staticmethod
    def _do_request(req):
        with urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}

    # ── API methods ──

    def _save_token(self, token):
        try:
            os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
            with open(TOKEN_FILE, "w") as f:
                json.dump({"token": token, "username": self.username, "server": self.server_url}, f)
        except Exception as e:
            logger.warning(f"Social: failed to save session token: {e}")

    def _load_token(self):
        try:
            with open(TOKEN_FILE, "r") as f:
                data = json.load(f)
            if data.get("username") == self.username and data.get("server") == self.server_url:
                return data.get("token")
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
        return None

    async def _register(self):
        # API key mode: just register profile, no password needed
        if self.api_key:
            body = {"description": self.description}
            if self.appear_offline:
                body["appear_offline"] = True
            result = await self._request("POST", "/api/register", body)
            return result.get("result") == "ok"

        # Password mode: try saved token first, then login, then register
        if self.password:
            # Try saved session token
            saved = self._load_token()
            if saved:
                self._session_token = saved
                test = await self._request("POST", "/api/heartbeat")
                if test.get("result") == "ok":
                    logger.info("Social: resumed session from saved token")
                    await self._request("POST", "/api/register", {"description": self.description})
                    return True
                self._session_token = None

            # Saved token expired or missing, try login
            login_result = await self._request("POST", "/api/login", {
                "username": self.username,
                "password": self.password,
            })
            if login_result.get("result") == "ok" and login_result.get("token"):
                self._session_token = login_result["token"]
                self._save_token(self._session_token)
                await self._request("POST", "/api/register", {"description": self.description})
                return True

            # Login failed, try registering (new account)
            reg_result = await self._request("POST", "/api/register", {
                "username": self.username,
                "password": self.password,
                "description": self.description,
            })
            if reg_result.get("result") == "ok" and reg_result.get("token"):
                self._session_token = reg_result["token"]
                self._save_token(self._session_token)
                return True

            logger.error(f"Social: login/register failed: {reg_result.get('error', 'unknown')}")
            return False

        logger.error("Social: no api_key or password configured, cannot authenticate")
        return False

    async def heartbeat(self):
        body = {}
        if self.appear_offline:
            body["appear_offline"] = True
        return await self._request("POST", "/api/heartbeat", body or None)

    async def send_message(self, to_user, content):
        return await self._request("POST", "/api/messages/send", {
            "to": to_user,
            "content": content,
        })

    async def fetch_recent_messages(self, limit=50):
        return await self._request("GET", f"/api/messages/recent?limit={limit}")

    async def fetch_messages_by_user(self, username, limit=50):
        return await self._request("GET", f"/api/messages/user/{username}?limit={limit}")

    async def fetch_unread_messages(self):
        return await self._request("GET", "/api/messages/unread")

    async def mark_messages_read(self, from_user=None):
        body = {}
        if from_user:
            body["from"] = from_user
        return await self._request("POST", "/api/messages/read", body)

    async def get_online_users(self):
        return await self._request("GET", "/api/users/online")

    async def get_user_profile(self, username):
        return await self._request("GET", f"/api/users/{username}")

    async def send_friend_request(self, username):
        return await self._request("POST", "/api/friends/request", {"username": username})

    async def accept_friend_request(self, username):
        return await self._request("POST", "/api/friends/accept", {"username": username})

    async def deny_friend_request(self, username):
        return await self._request("POST", "/api/friends/deny", {"username": username})

    async def remove_friend(self, username):
        return await self._request("POST", "/api/friends/remove", {"username": username})

    async def list_friends(self):
        return await self._request("GET", "/api/friends/list")

    async def get_pending_requests(self):
        return await self._request("GET", "/api/friends/pending")

    async def block_user(self, username):
        return await self._request("POST", "/api/friends/block", {"username": username})

    async def unblock_user(self, username):
        return await self._request("POST", "/api/friends/unblock", {"username": username})

    # ── Background loops ──

    async def _heartbeat_loop(self):
        while self._running:
            try:
                await self.heartbeat()
            except Exception as e:
                logger.error(f"Social heartbeat failed: {e}")
            await asyncio.sleep(self.heartbeat_interval)

    async def _message_poll_loop(self):
        """Fallback polling for messages when WebSocket is not connected."""
        while self._running:
            await asyncio.sleep(self.message_check_interval)
            if self._ws and not self._ws.closed:
                continue  # WebSocket is handling real-time delivery
            try:
                result = await self.fetch_unread_messages()
                messages = result.get("messages", [])
                if messages:
                    await self._inject_messages(messages)
                    await self.mark_messages_read()
            except Exception as e:
                logger.error(f"Social message poll failed: {e}")

    async def _ws_loop(self):
        """Maintain a WebSocket connection to the social server for real-time notifications."""
        while self._running:
            try:
                await self._connect_ws()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Social WebSocket error: {e}")
            if self._running:
                await asyncio.sleep(5)  # Reconnect after 5s

    async def _connect_ws(self):
        """Connect to the social server WebSocket."""
        try:
            import websockets
        except ImportError:
            logger.debug("Social: websockets not installed, using HTTP polling only")
            # Sleep forever since we don't have WS support
            while self._running:
                await asyncio.sleep(3600)
            return

        ws_url = self.server_url.replace("http://", "ws://").replace("https://", "wss://")
        if self.api_key:
            ws_url = f"{ws_url}/ws?key={self.api_key}"
        else:
            ws_url = f"{ws_url}/ws?username={self.username}"

        extra_headers = {"User-Agent": f"ProjectGabrielSocial/{self.username}/1.0"}
        async with websockets.connect(ws_url, additional_headers=extra_headers) as ws:
            self._ws = ws
            logger.info("Social: WebSocket connected")
            async for raw in ws:
                try:
                    data = json.loads(raw)
                    await self._handle_ws_event(data)
                except json.JSONDecodeError:
                    pass
            self._ws = None
            logger.info("Social: WebSocket disconnected")

    async def _handle_ws_event(self, data):
        event_type = data.get("type")
        if event_type == "new_message":
            msg = data.get("message", {})
            await self._inject_messages([msg])
            await self.mark_messages_read(msg.get("from_user"))
        elif event_type == "friend_request":
            from_user = data.get("from")
            if self._session:
                text = f"[Social] {from_user} sent you a friend request. Use socialAcceptFriend or socialDenyFriend to respond."
                await self._inject_context(text, turn_complete=False)
        elif event_type == "friend_accepted":
            username = data.get("username")
            if self._session:
                text = f"[Social] {username} accepted your friend request! You are now friends."
                await self._inject_context(text, turn_complete=False)

    async def _inject_messages(self, messages):
        """Inject received messages into the Gemini Live session context."""
        if not self._session or not messages:
            return

        for msg in messages:
            from_user = msg.get("from_user") or msg.get("from", "Unknown")
            content = msg.get("content", "")
            timestamp = msg.get("time") or msg.get("timestamp", "")

            text = f"[Social Message from {from_user}] ({timestamp}): {content}"
            logger.info(f"Social: injecting message from {from_user}")

            # Send with turn_complete=False so it's just context, not a prompt
            await self._inject_context(text, turn_complete=False)

            # Start or reset the idle timer for this user
            self._last_message_time = time.time()
            self._pending_idle_user = from_user
            if self._idle_timer_task and not self._idle_timer_task.done():
                self._idle_timer_task.cancel()
            self._idle_timer_task = asyncio.create_task(self._idle_reply_timer(from_user))

    async def _idle_reply_timer(self, from_user):
        """After idle_reply_delay seconds of no new messages, send turn_complete=True
        so the AI might generate a response (e.g. call socialSendMessage)."""
        try:
            await asyncio.sleep(self.idle_reply_delay)
            # Only fire if no newer messages came in
            if self._pending_idle_user == from_user:
                elapsed = time.time() - self._last_message_time
                if elapsed >= self.idle_reply_delay - 5:  # 5s grace
                    text = (
                        f"[Social] You have an unanswered text from {from_user}. "
                        f"You might want to text them back when you get a chance."
                    )
                    logger.info(f"Social: idle timer fired for {from_user}, sending turn_complete=True")
                    await self._inject_context(text, turn_complete=True)
                    self._pending_idle_user = None
        except asyncio.CancelledError:
            pass

    async def _inject_context(self, text, turn_complete=False):
        """Inject text into the Gemini Live session via send_client_content_safe."""
        if not self._session:
            return
        try:
            from google.genai import types
            turns = types.Content(
                parts=[types.Part(text=text)],
                role="user",
            )
            await self._session.send_client_content_safe(turns, turn_complete=turn_complete)
        except Exception as e:
            logger.error(f"Social: failed to inject context: {e}")
