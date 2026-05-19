import logging
from google.genai import types
from src.tools._base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool
class MappingTools(BaseTool):
    """Tools for the AI to use the world mapping + waypoint system.
    Lets the model save spots and walk back to them later via A*."""

    tool_key = "mapping"

    def _ms(self):
        ms = getattr(self.handler, "mapping_service", None)
        if ms is None:
            return None
        return ms

    def declarations(self, config=None):
        return [
            types.FunctionDeclaration(
                name="saveWaypoint",
                description=(
                    "Save the current avatar position as a named waypoint "
                    "in the current VRChat world so you can return to it "
                    "later via gotoWaypoint.\n"
                    "**Invocation Condition:** Call when someone asks you "
                    "to remember a spot, mark a location, or save where "
                    "you are right now."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "name": {
                            "type": "STRING",
                            "description": "Short label for the spot, eg 'couch' or 'mirror'.",
                        },
                        "note": {
                            "type": "STRING",
                            "description": "Optional note about the spot.",
                        },
                    },
                    "required": ["name"],
                },
            ),
            types.FunctionDeclaration(
                name="gotoWaypoint",
                description=(
                    "Walk to a previously saved waypoint in the current "
                    "VRChat world. The system pathfinds dynamically from "
                    "wherever you are right now using the live voxel map.\n"
                    "**Invocation Condition:** Call when asked to go to, "
                    "walk to, return to, or visit a named spot."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "name": {
                            "type": "STRING",
                            "description": "Name of the waypoint to walk to.",
                        },
                    },
                    "required": ["name"],
                },
            ),
            types.FunctionDeclaration(
                name="listWaypoints",
                description=(
                    "List every waypoint saved for the current VRChat world.\n"
                    "**Invocation Condition:** Call when asked what spots "
                    "you remember, or which waypoints exist."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="deleteWaypoint",
                description=(
                    "Forget a saved waypoint by name.\n"
                    "**Invocation Condition:** Call when asked to forget "
                    "or remove a saved spot."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                    },
                    "required": ["name"],
                },
            ),
            types.FunctionDeclaration(
                name="cancelWalk",
                description=(
                    "Stop walking to the current waypoint target.\n"
                    "**Invocation Condition:** Call when asked to stop, "
                    "halt, or cancel walking somewhere."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
        ]

    async def handle(self, name, args):
        ms = self._ms()
        if ms is None:
            return {"result": "error", "message": "mapping system not available"}

        if name == "saveWaypoint":
            wp_name = (args.get("name") or "").strip()
            if not wp_name:
                return {"result": "error", "message": "name required"}
            note = (args.get("note") or "").strip()
            try:
                wp = ms.add_waypoint(wp_name, note=note)
            except Exception as e:
                return {"result": "error", "message": str(e)}
            # ms.add_waypoint returns a dict (already serialized)
            return {"result": "ok", "waypoint": {
                "name": wp.get("name"),
                "x": wp.get("x"),
                "y": wp.get("y"),
                "z": wp.get("z"),
                "note": wp.get("note", ""),
            }}

        if name == "gotoWaypoint":
            wp_name = (args.get("name") or "").strip()
            if not wp_name:
                return {"result": "error", "message": "name required"}
            try:
                r = ms.goto_waypoint(wp_name)
            except RuntimeError as e:
                return {"result": "error", "message": str(e)}
            if not r.get("found"):
                return {"result": "error", "message": r.get("reason", "no path")}
            return {"result": "ok", "driving": True,
                    "cells": len(r.get("full") or []),
                    "turns": len(r.get("filtered") or [])}

        if name == "listWaypoints":
            try:
                items = ms.list_waypoints()
            except Exception as e:
                return {"result": "error", "message": str(e)}
            return {"result": "ok", "count": len(items),
                    "waypoints": [w.get("name") for w in items]}

        if name == "deleteWaypoint":
            wp_name = (args.get("name") or "").strip()
            if not wp_name:
                return {"result": "error", "message": "name required"}
            ok = ms.remove_waypoint(wp_name)
            return {"result": "ok" if ok else "not_found"}

        if name == "cancelWalk":
            ms.cancel_goto()
            return {"result": "ok"}

        return None
