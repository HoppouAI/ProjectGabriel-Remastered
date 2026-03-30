import yaml
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PERSONALITIES_FILE = Path("config/prompts/personalities.yml")


class PersonalityManager:
    def __init__(self, personalities_file=None):
        self._file = Path(personalities_file) if personalities_file else PERSONALITIES_FILE
        self.personalities = {}
        self.current = None
        self.history = []
        self._load()

    def _load(self):
        if self._file.exists():
            with open(self._file, "r", encoding="utf-8") as f:
                self.personalities = yaml.safe_load(f) or {}
            logger.info(f"Loaded {len(self.personalities)} personalities")

    def _save(self):
        with open(self._file, "w", encoding="utf-8") as f:
            yaml.dump(self.personalities, f, default_flow_style=False, allow_unicode=True)

    def list_personalities(self):
        result = []
        for pid, p in self.personalities.items():
            result.append({
                "id": pid,
                "name": p.get("name", pid),
                "description": p.get("description", ""),
                "enabled": p.get("enabled", True),
                "active": pid == self.current,
            })
        return {"personalities": result, "current": self.current}

    def switch(self, personality_id: str):
        if personality_id not in self.personalities:
            return {"error": f"Personality '{personality_id}' not found", "available": list(self.personalities.keys())}

        p = self.personalities[personality_id]
        if not p.get("enabled", True):
            return {"error": "This personality is disabled", "id": personality_id}

        self.history.append({
            "from": self.current,
            "to": personality_id,
            "time": datetime.now().isoformat(),
        })
        self.current = personality_id
        prompt = p.get("prompt", "")
        avatar_id = p.get("avatar_id", "")
        logger.info(f"Switched to personality: {personality_id}")
        result = {
            "result": "ok",
            "name": p.get("name", personality_id),
            "personality_prompt": f"Personality update - you are now {p.get('name', personality_id)}. {prompt}",
        }
        if avatar_id:
            result["avatar_id"] = avatar_id
        return result

    def get_current(self):
        if self.current and self.current in self.personalities:
            p = self.personalities[self.current]
            return {"id": self.current, "name": p.get("name"), "prompt": p.get("prompt", "")}
        return {"id": None, "name": "default"}

    def get_available_text(self):
        lines = ["Available personalities:"]
        for pid, p in self.personalities.items():
            if p.get("enabled", True):
                lines.append(f"- {pid}: {p.get('name', pid)} — {p.get('description', '')}")
        return "\n".join(lines)
