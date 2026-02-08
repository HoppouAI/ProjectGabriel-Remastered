import asyncio
import time
import logging
import threading
import queue
from pythonosc import udp_client
from pynput.keyboard import Key, Controller as KeyboardController

logger = logging.getLogger(__name__)

CHATBOX_CHAR_LIMIT = 144
CHATBOX_RATE_LIMIT = 1.27  # VRChat chatbox rate limit in seconds

# Keyboard controller for VRChat actions
_keyboard = KeyboardController()


class VRChatOSC:
    def __init__(self, config):
        self.config = config
        self.client = udp_client.SimpleUDPClient(config.osc_ip, config.osc_port)
        self._typing = False
        self._last_chatbox_time = 0
        self._chatbox_queue = queue.Queue()
        self._chatbox_thread = threading.Thread(target=self._chatbox_worker, daemon=True)
        self._chatbox_thread.start()

    def _chatbox_worker(self):
        """Background thread that sends chatbox messages respecting rate limit."""
        while True:
            try:
                text = self._chatbox_queue.get()
                
                # Rate limit enforcement
                now = time.time()
                elapsed = now - self._last_chatbox_time
                if elapsed < CHATBOX_RATE_LIMIT:
                    time.sleep(CHATBOX_RATE_LIMIT - elapsed)
                
                # Send the message
                self.client.send_message("/chatbox/input", [text, True, False])
                self._last_chatbox_time = time.time()
                
            except Exception as e:
                logger.error(f"Chatbox worker error: {e}")

    def set_typing(self, typing: bool):
        if self._typing != typing:
            self._typing = typing
            self.client.send_message("/chatbox/typing", typing)

    def send_chatbox(self, text: str):
        """Queue a chatbox message (non-blocking, sent by background thread)."""
        # Clear any pending messages and use the latest text
        try:
            while True:
                self._chatbox_queue.get_nowait()
        except queue.Empty:
            pass
        self._chatbox_queue.put(text)

    def send_chatbox_paginated(self, text: str) -> list[str]:
        if len(text) <= CHATBOX_CHAR_LIMIT:
            self.send_chatbox(text)
            return [text]
        pages = self._paginate(text)
        if pages:
            self.send_chatbox(pages[0])
        return pages

    def _paginate(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return [text[:CHATBOX_CHAR_LIMIT]]

        indicator_reserve = len(" (99/99)")
        usable = CHATBOX_CHAR_LIMIT - indicator_reserve

        chunks = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip() if current else word
            if len(test) > usable:
                if current:
                    chunks.append(current)
                    current = word
                else:
                    chunks.append(word[:usable])
                    current = ""
            else:
                current = test
        if current:
            chunks.append(current)

        total = len(chunks)
        return [f"{c} ({i + 1}/{total})" for i, c in enumerate(chunks)]

    def send_chatbox_direct(self, text: str):
        """Send chatbox message directly via the queue (for paginated display)."""
        self._chatbox_queue.put(text)

    async def display_pages(self, pages: list[str], delay: float = 3.0):
        # Ensure delay is at least the rate limit
        actual_delay = max(delay, CHATBOX_RATE_LIMIT)
        for page in pages:
            self.send_chatbox_direct(page)
            self.set_typing(True)
            await asyncio.sleep(actual_delay)

    def toggle_voice(self):
        self.client.send_message("/input/Voice", 1)
        time.sleep(0.05)
        self.client.send_message("/input/Voice", 0)

    def set_movement(self, forward: float = 0.0, horizontal: float = 0.0):
        self.client.send_message("/input/MoveForward", max(-1.0, min(1.0, forward)))
        self.client.send_message("/input/LookHorizontal", max(-1.0, min(1.0, horizontal)))

    def stop_movement(self):
        self.set_movement(0.0, 0.0)

    def toggle_crouch(self):
        """Toggle crouch in VRChat by pressing C key."""
        _keyboard.press('c')
        time.sleep(0.05)
        _keyboard.release('c')
        logger.info("Toggled crouch (C key)")

    def toggle_crawl(self):
        """Toggle crawl/prone in VRChat by pressing Z key."""
        _keyboard.press('z')
        time.sleep(0.05)
        _keyboard.release('z')
        logger.info("Toggled crawl (Z key)")

    # Movement methods for person tracking (reference implementation style)
    def _move_forward(self):
        """Start moving forward."""
        self.client.send_message("/input/MoveForward", 1)

    def _stop_forward(self):
        """Stop moving forward."""
        self.client.send_message("/input/MoveForward", 0)

    def _move_backward(self):
        """Start moving backward."""
        self.client.send_message("/input/MoveBackward", 1)

    def _stop_backward(self):
        """Stop moving backward."""
        self.client.send_message("/input/MoveBackward", 0)

    async def rotate_left(self, steps: int = 1):
        """Rotate left with timed key press."""
        for _ in range(steps):
            self.client.send_message("/input/LookLeft", 1)
            await asyncio.sleep(0.1)
            self.client.send_message("/input/LookLeft", 0)

    async def rotate_right(self, steps: int = 1):
        """Rotate right with timed key press."""
        for _ in range(steps):
            self.client.send_message("/input/LookRight", 1)
            await asyncio.sleep(0.1)
            self.client.send_message("/input/LookRight", 0)

    # Manual movement methods for AI control
    def start_move(self, direction: str):
        """Start moving in a direction (forward, backward, left, right)."""
        if direction == "forward":
            self.client.send_message("/input/MoveForward", 1)
        elif direction == "backward":
            self.client.send_message("/input/MoveBackward", 1)
        elif direction == "left":
            self.client.send_message("/input/MoveLeft", 1)
        elif direction == "right":
            self.client.send_message("/input/MoveRight", 1)
        logger.info(f"Started moving {direction}")

    def stop_all_movement(self):
        """Stop all movement."""
        self.client.send_message("/input/MoveForward", 0)
        self.client.send_message("/input/MoveBackward", 0)
        self.client.send_message("/input/MoveLeft", 0)
        self.client.send_message("/input/MoveRight", 0)
        logger.info("Stopped all movement")

    def jump(self):
        """Make the avatar jump."""
        self.client.send_message("/input/Jump", 1)
        time.sleep(0.05)
        self.client.send_message("/input/Jump", 0)
        logger.info("Jumped")
