import math

from google.genai import types

from src.tools._base import BaseTool, register_tool

SUPPORTED_MIN_METERS = 0.1
SUPPORTED_MAX_METERS = 100.0


@register_tool
class AvatarScalingTools(BaseTool):
    tool_key = "avatar_scaling"

    def declarations(self, config=None):
        return [
            types.FunctionDeclaration(
                name="vrchatSetAvatarScale",
                description=(
                    "Set the avatar's VRChat height/scale in meters. Clamps to the officially supported 0.1m to 100m range.\n"
                    "**Invocation Condition:** Call when asked "
                    "to become a specific height, resize to a meter value, get bigger or smaller to an exact size, or set avatar scale."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "height_meters": {"type": "NUMBER", "description": "Target avatar height in meters. Supported range is 0.1 to 100."},
                    },
                    "required": ["height_meters"],
                },
            ),
            types.FunctionDeclaration(
                name="vrchatAdjustAvatarScale",
                description=(
                    "Increase or decrease the avatar's VRChat height/scale relative to the latest known value. "
                    "Clamps to 0.1m to 100m.\n**Invocation Condition:** Call when "
                    "asked to scale up, scale down, grow, shrink, become taller, or become smaller by an amount in meters."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "direction": {"type": "STRING", "description": "Scale direction: 'up' to increase height, 'down' to decrease height."},
                        "amount_meters": {"type": "NUMBER", "description": "How many meters to add or subtract from the latest known height."},
                    },
                    "required": ["direction", "amount_meters"],
                },
            ),
            types.FunctionDeclaration(
                name="vrchatGetAvatarScale",
                description=(
                    "Get the latest avatar scaling status, including height in meters, world min/max values, "
                    "and whether scaling appears allowed.\n**Invocation Condition:** Call before relative "
                    "scaling if the current avatar height is unknown, or when asked how tall or scaled the avatar currently is."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        if name == "vrchatSetAvatarScale":
            height, error = self._read_finite_number(args.get("height_meters"), "height_meters")
            if error:
                return error
            return self._set_height(height)
        elif name == "vrchatAdjustAvatarScale":
            direction = str(args.get("direction", "")).strip().lower()
            if direction not in ("up", "down"):
                return {"result": "error", "message": "direction must be 'up' or 'down'"}
            amount, error = self._read_finite_number(args.get("amount_meters"), "amount_meters")
            if error:
                return error
            amount = abs(amount)
            if amount <= 0:
                return {"result": "error", "message": "amount_meters must be greater than 0"}
            base = self._latest_height()
            status = self.osc.get_avatar_scaling_status()
            if base is None:
                return {
                    "result": "error",
                    "message": "Current avatar height is unknown. Use vrchatSetAvatarScale with an absolute height in meters first.",
                    "status": status,
                }
            target = base + amount if direction == "up" else base - amount
            result = self._set_height(target)
            result["base_height_meters"] = base
            result["direction"] = direction
            result["amount_meters"] = amount
            result["target_height_meters"] = target
            return result
        elif name == "vrchatGetAvatarScale":
            return {"result": "ok", "status": self.osc.get_avatar_scaling_status()}
        return None

    @staticmethod
    def _read_finite_number(value, field_name: str):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None, {"result": "error", "message": f"{field_name} must be a number"}
        if not math.isfinite(number):
            return None, {"result": "error", "message": f"{field_name} must be finite"}
        return number, None

    def _latest_height(self):
        actual = getattr(self.osc, "avatar_eye_height_meters", None)
        if actual is not None:
            return actual
        return getattr(self.osc, "last_requested_avatar_eye_height_meters", None)

    def _set_height(self, height_meters: float):
        result = self.osc.set_avatar_eye_height(height_meters)
        status = self.osc.get_avatar_scaling_status()
        payload = {"result": "ok", **result, "status": status}
        if status.get("scaling_allowed") is False:
            payload["warning"] = "VRChat reports avatar scaling as disabled in this world, so the request may be ignored."
        elif status.get("eye_height_min_meters") is not None or status.get("eye_height_max_meters") is not None:
            payload["note"] = "World min and max values are menu limits. Udon may still override the result."
        return payload
