import logging
from datetime import datetime
import pytz
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class TimeTools(BaseTool):
    tool_key = "time"

    def declarations(self, config=None):
        return [
            types.FunctionDeclaration(
                name="getCurrentTime",
                description="Get the current time in 12-hour format (HH:MM AM/PM) for the local system timezone.\n**Invocation Condition:** Call when asked what time it is, or you need to know the current time.",
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="getTimeInZone",
                description="Get the current time in 12-hour format for a specific timezone. Pass the timezone name (e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo', 'Australia/Sydney').\n**Invocation Condition:** Call when asked about time in a specific timezone or location.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "timezone": {
                            "type": "STRING",
                            "description": "IANA timezone name (e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo'). Common zones: America/New_York, America/Chicago, America/Denver, America/Los_Angeles, America/Anchorage, Pacific/Honolulu, Europe/London, Europe/Paris, Europe/Berlin, Asia/Tokyo, Asia/Shanghai, Asia/Hong_Kong, Asia/Singapore, Asia/Dubai, Asia/Bangkok, Asia/Kolkata, Australia/Sydney, Australia/Melbourne, Australia/Brisbane, Pacific/Auckland",
                        }
                    },
                    "required": ["timezone"],
                },
            ),
        ]

    async def handle(self, name, args):
        if name == "getCurrentTime":
            return self._get_current_time()
        elif name == "getTimeInZone":
            tz = args.get("timezone", "").strip()
            if not tz:
                return {"result": "error", "message": "timezone parameter required"}
            return self._get_time_in_zone(tz)
        return None

    def _get_current_time(self):
        now = datetime.now()
        time_str = now.strftime("%I:%M %p").lstrip("0")
        date_str = now.strftime("%A, %B %d, %Y")
        
        try:
            tz_name = datetime.now(pytz.timezone("UTC")).astimezone().tzname()
        except Exception:
            tz_name = "Local"
        
        return {
            "result": "ok",
            "time": time_str,
            "date": date_str,
            "timezone": tz_name,
            "full": f"{time_str} - {date_str}",
        }

    def _get_time_in_zone(self, timezone):
        try:
            tz = pytz.timezone(timezone)
            now = datetime.now(tz)
            time_str = now.strftime("%I:%M %p").lstrip("0")
            date_str = now.strftime("%A, %B %d, %Y")
            
            return {
                "result": "ok",
                "time": time_str,
                "date": date_str,
                "timezone": timezone,
                "full": f"{time_str} - {date_str}",
            }
        except pytz.exceptions.UnknownTimeZoneError:
            available = ", ".join(sorted(pytz.common_timezones)[:20])
            return {
                "result": "error",
                "message": f"Unknown timezone: {timezone}. Examples: America/New_York, Europe/London, Asia/Tokyo, Australia/Sydney.",
                "hint": available + "...",
            }
        except Exception as e:
            return {
                "result": "error",
                "message": f"Failed to get time for {timezone}: {str(e)}",
            }
