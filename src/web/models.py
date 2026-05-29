"""Pydantic request bodies for the control panel API."""
from __future__ import annotations
from pydantic import BaseModel


class TextInput(BaseModel):
    text: str


class MusicInput(BaseModel):
    filename: str


class VolumeInput(BaseModel):
    volume: float


class PersonalityInput(BaseModel):
    personality: str


class EmotionInput(BaseModel):
    emotion: str


class MemoryCreateInput(BaseModel):
    key: str | None = None
    content: str
    category: str = "general"
    memory_type: str = "long_term"
    tags: list[str] | None = None


class MemoryUpdateInput(BaseModel):
    content: str | None = None
    category: str | None = None
    memory_type: str | None = None
    tags: list[str] | None = None


class MemoryPinInput(BaseModel):
    pin: bool = True


class ModerationInput(BaseModel):
    moderated: str
    type: str


class MappingStartInput(BaseModel):
    explore: bool = False


class MappingExploreInput(BaseModel):
    enabled: bool


class MappingManualInput(BaseModel):
    enabled: bool


class PathfindInput(BaseModel):
    x: float
    y: float = 0.0
    z: float
    waypoint: str | None = None


class WaypointCreateInput(BaseModel):
    name: str
    note: str = ""


class MappingSettingsIn(BaseModel):
    tick_hz: float | None = None
    force_run: bool | None = None
    manual_wall_distance: float | None = None
    manual_wall_ratio: float | None = None


class GotoIn(BaseModel):
    waypoint: str | None = None
    x: float | None = None
    y: float | None = None
    z: float | None = None


class CellEditIn(BaseModel):
    sx: int
    sy: int
    sz: int
    kind: str


class CellsBulkEditIn(BaseModel):
    cells: list[list[int]]
    kind: str


class CleanupStraysIn(BaseModel):
    min_component_size: int = 8
    dry_run: bool = False


class CleanupJumpArtifactsIn(BaseModel):
    dry_run: bool = False


class YoloConfigInput(BaseModel):
    config: dict
