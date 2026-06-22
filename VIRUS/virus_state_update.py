"""
Reusable VIRUS state-update adapter for ZSON projects.

This module intentionally stops at internal state updates. It never overrides
planner or action outputs, so downstream policies still act as pi(Mt+1, T).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Optional, Sequence

import numpy as np


GridPoint = tuple[int, int]


@dataclass
class VirusStateUpdateConfig:
    semantic_keys: Sequence[str] = field(default_factory=lambda: (
        "carefully",
        "quietly",
        "cautiously",
        "alert",
        "attentive",
    ))
    vmax: float = 10.0
    sigma_base: float = 1.5
    eta: float = 0.5
    xi: float = 0.1
    vlimit: float = 1.0
    connectivity: int = 8
    max_graph_distance: Optional[float] = None


class VirusStateUpdater:
    """TRAP + AH + GSU adapter that can be mounted at a ZSON state update hook."""

    def __init__(self, cfg: Optional[VirusStateUpdateConfig] = None):
        self.cfg = cfg or VirusStateUpdateConfig()

    def detect_semantic_key(self, instruction: str) -> tuple[bool, Optional[str]]:
        text = (instruction or "").lower()
        for key in self.cfg.semantic_keys:
            if key.lower() in text:
                return True, key
        return False, None

    def adaptive_sigma(
        self,
        velocity: Optional[np.ndarray] = None,
        acceleration: Optional[np.ndarray] = None,
    ) -> float:
        velocity_norm = float(np.linalg.norm(velocity)) if velocity is not None else 0.0
        acceleration_norm = float(np.linalg.norm(acceleration)) if acceleration is not None else 0.0
        vlimit = max(float(self.cfg.vlimit), 1e-6)
        beta = 1.0 + self.cfg.eta * velocity_norm / vlimit + self.cfg.xi * acceleration_norm
        return max(float(self.cfg.sigma_base) * beta, 1e-6)

    def build_viral_field(
        self,
        traversable_mask: np.ndarray,
        anchor_yx: GridPoint,
        velocity: Optional[np.ndarray] = None,
        acceleration: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict]:
        mask = np.asarray(traversable_mask, dtype=bool)
        anchor = self._snap_anchor(mask, anchor_yx)
        distances = self.graph_distances(mask, anchor)
        sigma = self.adaptive_sigma(velocity, acceleration)

        field = np.zeros(mask.shape, dtype=np.float32)
        reachable = np.isfinite(distances)
        if self.cfg.max_graph_distance is not None:
            reachable &= distances <= self.cfg.max_graph_distance
        field[reachable] = self.cfg.vmax * np.exp(-(distances[reachable] ** 2) / (2.0 * sigma ** 2))
        field[~mask] = 0.0

        metadata = {
            "anchor_yx": anchor,
            "sigma": sigma,
            "field_max": float(field.max()) if field.size else 0.0,
            "reachable_nodes": int(reachable.sum()),
        }
        return field, metadata

    def gsu_update(self, percept_state: np.ndarray, viral_field: np.ndarray) -> np.ndarray:
        aligned_field = self._align_field_to_state(viral_field, percept_state)
        return np.maximum(percept_state, aligned_field)

    def update_state(
        self,
        percept_state: np.ndarray,
        traversable_mask: np.ndarray,
        visual_active: bool,
        anchor_yx: Optional[GridPoint],
        instruction: str,
        velocity: Optional[np.ndarray] = None,
        acceleration: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict]:
        semantic_active, trigger_word = self.detect_semantic_key(instruction)
        activated = bool(visual_active and semantic_active and anchor_yx is not None)
        metadata = {
            "activated": activated,
            "visual_active": bool(visual_active),
            "semantic_active": semantic_active,
            "trigger_word": trigger_word,
        }

        if not activated:
            return percept_state, metadata

        viral_field, field_metadata = self.build_viral_field(
            traversable_mask,
            anchor_yx,
            velocity=velocity,
            acceleration=acceleration,
        )
        metadata.update(field_metadata)
        return self.gsu_update(percept_state, viral_field), metadata

    def graph_distances(self, traversable_mask: np.ndarray, anchor_yx: GridPoint) -> np.ndarray:
        mask = np.asarray(traversable_mask, dtype=bool)
        height, width = mask.shape
        distances = np.full((height, width), np.inf, dtype=np.float32)
        ay, ax = anchor_yx
        if not (0 <= ay < height and 0 <= ax < width and mask[ay, ax]):
            return distances

        distances[ay, ax] = 0.0
        heap: list[tuple[float, int, int]] = [(0.0, ay, ax)]
        while heap:
            dist, y, x = heapq.heappop(heap)
            if dist > distances[y, x]:
                continue
            if self.cfg.max_graph_distance is not None and dist > self.cfg.max_graph_distance:
                continue
            for ny, nx, weight in self._neighbors(y, x, height, width, self.cfg.connectivity):
                if not mask[ny, nx]:
                    continue
                next_dist = dist + weight
                if self.cfg.max_graph_distance is not None and next_dist > self.cfg.max_graph_distance:
                    continue
                if next_dist < distances[ny, nx]:
                    distances[ny, nx] = next_dist
                    heapq.heappush(heap, (next_dist, ny, nx))
        return distances

    @staticmethod
    def _neighbors(y: int, x: int, height: int, width: int, connectivity: int):
        offsets = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
        if connectivity == 8:
            diag = float(np.sqrt(2.0))
            offsets.extend([(-1, -1, diag), (-1, 1, diag), (1, -1, diag), (1, 1, diag)])
        for dy, dx, weight in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                yield ny, nx, weight

    @staticmethod
    def _snap_anchor(mask: np.ndarray, anchor_yx: GridPoint) -> GridPoint:
        y, x = anchor_yx
        height, width = mask.shape
        y = int(np.clip(y, 0, height - 1))
        x = int(np.clip(x, 0, width - 1))
        if mask[y, x]:
            return y, x

        traversable = np.argwhere(mask)
        if traversable.size == 0:
            return y, x
        distances = np.sum((traversable - np.array([y, x])) ** 2, axis=1)
        nearest = traversable[int(np.argmin(distances))]
        return int(nearest[0]), int(nearest[1])

    @staticmethod
    def _align_field_to_state(viral_field: np.ndarray, percept_state: np.ndarray) -> np.ndarray:
        field = np.asarray(viral_field, dtype=percept_state.dtype)
        if percept_state.ndim == field.ndim:
            return field
        if percept_state.ndim == 3 and field.ndim == 2:
            return np.repeat(field[:, :, None], percept_state.shape[2], axis=2)
        raise ValueError(
            f"Cannot align viral field shape {field.shape} to state shape {percept_state.shape}"
        )
