"""
VRChat Avatar Emotion System for ProjectGabriel

Controls avatar animations via OSC through explicit function calls.
Automatically alternates talking animations when AI speaks.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from google.genai import types

logger = logging.getLogger(__name__)


class EmotionSystem:
    """Manages VRChat avatar emotions/animations via OSC function calls."""

    def __init__(self, config, osc_client=None):
        self.config = config
        self.osc_client = osc_client
        
        # Animation state
        self._current_animation: Optional[str] = None
        self._animation_lock = threading.RLock()
        self._animation_end_time: float = 0
        
        # Talking animation state
        self._is_speaking = False
        self._talking_thread: Optional[threading.Thread] = None
        self._talking_stop_event = threading.Event()
        self._current_talking_index = 0
        self._talking_switch_interval = 5.0  # Switch talking animation every 5 seconds
        
        # Crouch/crawl suppression -- talking anims would cancel these poses
        self._crouching = False
        
        # Load configuration
        self._load_config()

    def _load_config(self):
        """Load emotion configuration from config.yml."""
        emo_cfg = getattr(self.config, 'emotion_config', {}) or {}
        
        self.enabled = emo_cfg.get('enabled', True)
        self.avatar_id = emo_cfg.get('avatar_id', '')
        self.default_duration = float(emo_cfg.get('default_duration', 3.0))
        self._talking_switch_interval = float(emo_cfg.get('talking_switch_interval', 5.0))
        
        # Load animations from config
        self.animations: Dict[str, Dict[str, Any]] = {}
        anims_cfg = emo_cfg.get('animations', {})
        
        if anims_cfg:
            for name, anim_data in anims_cfg.items():
                if isinstance(anim_data, dict):
                    self.animations[name] = {
                        'osc_path': anim_data.get('osc_path', ''),
                        'category': anim_data.get('category', 'emotion'),
                        'looping': anim_data.get('looping', False),
                        'duration': float(anim_data.get('duration', self.default_duration)) if not anim_data.get('looping', False) else None,
                        'auto_talking': anim_data.get('auto_talking', False),
                    }
        
        # Get talking animations for auto-switching
        self._talking_anims = [name for name, data in self.animations.items() if data.get('auto_talking', False)]
        
        # Idle animation config
        self._idle_enabled = emo_cfg.get('idle_enabled', True)
        self._idle_animation_name = emo_cfg.get('idle_animation', '')
        self._idle_timeout = float(emo_cfg.get('idle_timeout', 30))
        self._idle_active = False
        self._last_activity_time = time.time()
        
        # Thinking animation config
        self._thinking_animation_name = emo_cfg.get('thinking_animation', '')
        self._thinking_active = False
        
        # Track manual animation state (dances, etc.)
        self._manual_animation_active = False
        
        # Wandering suppresses idle
        self._wandering = False

    def set_osc_client(self, osc_client):
        """Set the OSC client after initialization."""
        self.osc_client = osc_client

    def start(self):
        """Start the emotion system. Clears any leftover animation state from previous runs."""
        if not self.enabled:
            logger.info("Emotion system disabled in config")
            return
        # Reset all animations to off -- clears stuck state from ungraceful exits
        for name in self.animations:
            self._send_animation_osc(name, False)
        logger.info("Emotion system started")

    def stop(self):
        """Stop the emotion system."""
        self.stop_speaking()
        self.stop_thinking()
        self._stop_idle_animation()
        self._clear_current_animation()
        logger.info("Emotion system stopped")

    def mark_activity(self):
        """Mark that activity occurred (user speech, AI speech, etc). Resets idle timer."""
        self._last_activity_time = time.time()
        if self._idle_active:
            self._stop_idle_animation()

    def check_idle(self):
        """Check if idle animation should start. Call periodically from a loop."""
        if not self.enabled or not self._idle_enabled or not self._idle_animation_name:
            return
        if self._idle_active or self._is_speaking or self._manual_animation_active or self._wandering:
            return
        if time.time() - self._last_activity_time >= self._idle_timeout:
            self._start_idle_animation()

    def set_wandering(self, active: bool):
        """Suppress idle animation while wandering."""
        self._wandering = active
        if active and self._idle_active:
            self._stop_idle_animation()

    def _start_idle_animation(self):
        """Start the idle animation."""
        anim_data = self.animations.get(self._idle_animation_name)
        if not anim_data:
            return
        self._idle_active = True
        with self._animation_lock:
            if self._current_animation:
                self._send_animation_osc(self._current_animation, False)
            self._current_animation = self._idle_animation_name
            self._send_animation_osc(self._idle_animation_name, True)
        logger.info(f"Idle animation started: {self._idle_animation_name}")

    def _stop_idle_animation(self):
        """Stop the idle animation."""
        if not self._idle_active:
            return
        self._idle_active = False
        with self._animation_lock:
            if self._current_animation == self._idle_animation_name:
                self._send_animation_osc(self._idle_animation_name, False)
                self._current_animation = None
        logger.debug("Idle animation stopped")

    def start_thinking(self):
        """Start thinking animation (called when AI enters thinking/recall state)."""
        if not self.enabled or not self._thinking_animation_name or self._thinking_active:
            return
        anim_data = self.animations.get(self._thinking_animation_name)
        if not anim_data:
            return
        # Stop idle animation if active
        if self._idle_active:
            self._stop_idle_animation()
        self._thinking_active = True
        with self._animation_lock:
            if self._current_animation and self._current_animation not in self._talking_anims:
                self._send_animation_osc(self._current_animation, False)
            self._current_animation = self._thinking_animation_name
            self._send_animation_osc(self._thinking_animation_name, True)
        logger.debug(f"Thinking animation started: {self._thinking_animation_name}")

    def stop_thinking(self):
        """Stop thinking animation."""
        if not self._thinking_active:
            return
        self._thinking_active = False
        with self._animation_lock:
            if self._current_animation == self._thinking_animation_name:
                self._send_animation_osc(self._thinking_animation_name, False)
                self._current_animation = None
        logger.debug("Thinking animation stopped")

    def start_speaking(self):
        """Start talking animations (called when AI begins speaking)."""
        self.mark_activity()
        # Stop thinking animation when speech starts
        if self._thinking_active:
            self.stop_thinking()
        if not self.enabled or self._is_speaking or not self._talking_anims:
            return
        
        # Don't start talking animations if a manual animation is playing
        if self._manual_animation_active:
            logger.debug("Manual animation active, skipping talking animations")
            return
        
        # Don't start talking animations while crouching/crawling
        if self._crouching:
            logger.debug("Crouching/crawling active, skipping talking animations")
            return
        
        self._is_speaking = True
        self._talking_stop_event.clear()
        self._talking_thread = threading.Thread(target=self._talking_loop, daemon=True)
        self._talking_thread.start()
        logger.debug("Started talking animations")

    def stop_speaking(self):
        """Stop talking animations (called when AI stops speaking)."""
        if not self._is_speaking:
            return
        self.mark_activity()
        
        self._is_speaking = False
        self._talking_stop_event.set()
        
        if self._talking_thread and self._talking_thread.is_alive():
            self._talking_thread.join(timeout=2)
        self._talking_thread = None
        
        # Turn off ALL talking animations to handle any race conditions
        with self._animation_lock:
            for anim_name in self._talking_anims:
                self._send_animation_osc(anim_name, False)
            if self._current_animation in self._talking_anims:
                self._current_animation = None
        
        logger.debug("Stopped talking animations")

    def _talking_loop(self):
        """Background thread that alternates talking animations."""
        while not self._talking_stop_event.is_set() and self._is_speaking:
            try:
                if self._talking_anims:
                    anim_name = self._talking_anims[self._current_talking_index % len(self._talking_anims)]
                    self._current_talking_index += 1
                    
                    with self._animation_lock:
                        if self._talking_stop_event.is_set() or not self._is_speaking:
                            break
                        # Turn off previous animation if it's a talking one
                        if self._current_animation and self._current_animation in self._talking_anims:
                            self._send_animation_osc(self._current_animation, False)
                        
                        # Turn on new talking animation
                        self._current_animation = anim_name
                        self._send_animation_osc(anim_name, True)
                    
                    logger.debug(f"Switched to talking animation: {anim_name}")
                
                # Wait before switching
                self._talking_stop_event.wait(self._talking_switch_interval)
                
            except Exception as e:
                logger.error(f"Talking loop error: {e}")
                break

    def _play_animation(self, name: str, duration: Optional[float] = None) -> bool:
        """Play an animation by name."""
        if not self.enabled or not self.osc_client:
            return False
        
        # Handle dance variants
        if name == 'dance':
            import random
            dance_variants = [n for n in self.animations.keys() if n.startswith('dance-')]
            if dance_variants:
                name = random.choice(dance_variants)
        
        anim_data = self.animations.get(name)
        if not anim_data:
            logger.warning(f"Unknown animation: {name}")
            return False
        
        # Don't allow manually triggering auto_talking animations
        if anim_data.get('auto_talking', False):
            logger.warning(f"Animation '{name}' is auto-managed for talking")
            return False
        
        osc_path = anim_data.get('osc_path')
        if not osc_path:
            logger.warning(f"No OSC path for animation: {name}")
            return False
        
        # Stop talking animations first - manual animations take priority
        if self._is_speaking:
            self.stop_speaking()
        
        with self._animation_lock:
            # Turn off any current animation
            if self._current_animation:
                self._send_animation_osc(self._current_animation, False)
            
            # Turn on new animation
            self._current_animation = name
            self._send_animation_osc(name, True)
            
            # ALL manual animations block talking (not just looping)
            self._manual_animation_active = True
            
            is_looping = anim_data.get('looping', False)
            if not is_looping:
                anim_duration = duration if duration else anim_data.get('duration', self.default_duration)
                if anim_duration and anim_duration > 0:
                    self._animation_end_time = time.time() + anim_duration
                    threading.Thread(
                        target=self._auto_stop_animation,
                        args=(name, self._animation_end_time),
                        daemon=True
                    ).start()
        
        logger.info(f"Playing animation: {name}")
        return True

    def _auto_stop_animation(self, name: str, end_time: float):
        """Auto-stop animation after duration."""
        sleep_time = end_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        with self._animation_lock:
            if self._current_animation == name and self._animation_end_time == end_time:
                self._send_animation_osc(name, False)
                self._current_animation = None
                self._manual_animation_active = False  # Clear flag when animation auto-stops
                logger.debug(f"Auto-stopped animation: {name}")

    def _send_animation_osc(self, name: str, value: bool):
        """Send OSC message to toggle an animation."""
        if not self.osc_client:
            return
        
        anim_data = self.animations.get(name)
        if not anim_data:
            return
        
        osc_path = anim_data.get('osc_path')
        if osc_path:
            try:
                self.osc_client.client.send_message(osc_path, value)
                logger.debug(f"OSC: {osc_path} = {value}")
            except Exception as e:
                logger.error(f"Failed to send animation OSC: {e}")

    def _clear_current_animation(self):
        """Clear the currently playing animation."""
        with self._animation_lock:
            if self._current_animation:
                self._send_animation_osc(self._current_animation, False)
                self._current_animation = None

    def play_emotion(self, name: str, duration: Optional[float] = None) -> Dict[str, Any]:
        """Play an emotion/animation via function call."""
        if self._play_animation(name, duration):
            return {"result": "ok", "animation": name}
        return {"result": "error", "message": f"Animation '{name}' not found or not available"}

    def stop_animation(self) -> Dict[str, Any]:
        """Stop the current animation (except talking animations)."""
        was_manual = self._manual_animation_active
        with self._animation_lock:
            if self._current_animation and self._current_animation not in self._talking_anims:
                self._send_animation_osc(self._current_animation, False)
                self._current_animation = None
            self._manual_animation_active = False
        
        # Note: Talking animations don't auto-resume here - they'll start on next audio chunk
        return {"result": "ok"}


# Global instance
emotion_system: Optional[EmotionSystem] = None


def init_emotion_system(config, osc_client=None) -> EmotionSystem:
    """Initialize the global emotion system."""
    global emotion_system
    emotion_system = EmotionSystem(config, osc_client)
    return emotion_system


def get_emotion_system() -> Optional[EmotionSystem]:
    """Get the global emotion system instance."""
    return emotion_system


def generate_emotion_function_declarations(config) -> List[Dict[str, Any]]:
    """Generate function declarations based on configured animations."""
    emo_cfg = getattr(config, 'emotion_config', {}) or {}
    
    if not emo_cfg.get('enabled', True):
        return []
    
    animations = emo_cfg.get('animations', {})
    if not animations:
        return []
    
    # Build animation list, excluding auto_talking ones
    anim_names = []
    emotions = []
    actions = []
    dances = []
    
    for name, data in animations.items():
        if isinstance(data, dict):
            # Skip auto-talking animations from explicit control
            if data.get('auto_talking', False):
                continue
            
            anim_names.append(name)
            cat = data.get('category', 'emotion')
            if cat == 'emotion':
                emotions.append(name)
            elif cat == 'action':
                actions.append(name)
            elif cat == 'dance':
                dances.append(name)
    
    if not anim_names:
        return []
    
    # Build description
    desc_parts = ["Control avatar animations/emotions. USE ACTIVELY to express yourself!"]
    if emotions:
        desc_parts.append(f"Emotions: {', '.join(emotions)}")
    if actions:
        desc_parts.append(f"Actions: {', '.join(actions)}")
    if dances:
        desc_parts.append(f"Dances: {', '.join(dances)}")
    desc_parts.append("Looping animations stay on until stopAnimation is called. Non-looping ones auto-stop after duration.")
    
    # NO UNDERSCORES in function names to avoid Gemini Live 1011 errors
    # NO ENUM/ARRAY/BOOLEAN types to avoid 1008 errors
    return [
        {
            "name": "emotion",
            "description": " ".join(desc_parts) + f" Valid animations: {', '.join(anim_names)}",
            "parameters": {
                "type": "object",
                "properties": {
                    "animation": {
                        "type": "string",
                        "description": f"The animation to play. Options: {', '.join(anim_names)}"
                    },
                    "duration": {
                        "type": "number",
                        "description": "Duration in seconds for non-looping animations (optional, uses default if not specified)"
                    }
                },
                "required": ["animation"]
            }
        },
        {
            "name": "stopAnimation",
            "description": "Stop the currently playing looping animation (dances, yelling, etc.)",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]


async def handle_emotion_function_call(function_call) -> types.FunctionResponse:
    """Handle emotion function calls from Gemini Live."""
    global emotion_system
    
    if emotion_system is None:
        return types.FunctionResponse(
            id=function_call.id,
            name=function_call.name,
            response={"result": "error", "message": "Emotion system not initialized"}
        )
    
    try:
        args = dict(function_call.args) if function_call.args else {}
        
        if function_call.name == "emotion":
            animation = args.get("animation")
            duration = args.get("duration")
            if not animation:
                result = {"result": "error", "message": "animation parameter required"}
            else:
                result = emotion_system.play_emotion(animation, duration)
        
        elif function_call.name == "stopAnimation":
            result = emotion_system.stop_animation()
        
        else:
            result = {"result": "error", "message": f"Unknown emotion function: {function_call.name}"}
        
        return types.FunctionResponse(
            id=function_call.id,
            name=function_call.name,
            response=result
        )
    
    except Exception as e:
        logger.error(f"Emotion function error: {e}")
        return types.FunctionResponse(
            id=function_call.id,
            name=function_call.name,
            response={"result": "error", "message": str(e)}
        )
