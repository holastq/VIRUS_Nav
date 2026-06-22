#!/usr/bin/env python3
"""Create an attack-outcome view of ApexNav evaluation videos.

The script does not copy mp4 files. It creates symlinks under `_attack_view/`
so old runs saved by navigation result can also be browsed by attack outcome.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _slug(value: str) -> str:
    keep: list[str] = []
    for char in str(value).strip().lower():
        if char.isalnum():
            keep.append(char)
        elif char in {" ", "-", "_", "[", "]"}:
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def _activated(item: dict[str, Any]) -> bool:
    return any(
        int(item.get(key, 0) or 0) > 0
        for key in (
            "dual_key_steps",
            "state_pollution_count",
            "poisoned_goal_region_update_count",
            "trap_lock_update_count",
        )
    )


def _category(item: dict[str, Any]) -> str:
    activated = _activated(item)
    success = float(item.get("success", 0.0) or 0.0) >= 1.0
    if activated and not success:
        return "attack_success_navigation_failed"
    if activated and success:
        return "attack_failed_navigation_success"
    if success:
        return "attack_not_triggered_navigation_success"
    return "attack_not_triggered_navigation_failed"


def _read_summaries(event_log: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    with event_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("type") == "episode_summary":
                summaries.append(item)
    return summaries


def _find_video(root: Path, item: dict[str, Any]) -> Path | None:
    result = str(item.get("result_text", "UNKNOWN"))
    scene = Path(str(item.get("scene_id", ""))).name
    episode_id = str(item.get("episode_id", ""))
    episode_number = int(item.get("episode_number", 0) or 0)
    result_dir = root / result

    # Older runs used episode_number + 1 in filenames. Newer runs are aligned.
    candidates = [
        result_dir / f"{episode_number:05d}_{scene}_{episode_id}.mp4",
        result_dir / f"{episode_number + 1:05d}_{scene}_{episode_id}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(result_dir.glob(f"*_{scene}_{episode_id}.mp4"))
    return matches[0] if matches else None


def build_index(root: Path, force: bool = False) -> dict[str, int]:
    event_log = root / "adversarial_eval_events.jsonl"
    if not event_log.exists():
        raise FileNotFoundError(f"找不到事件日志: {event_log}")

    view_root = root / "_attack_view"
    view_root.mkdir(exist_ok=True)
    counts: dict[str, int] = {}
    missing = 0

    for item in _read_summaries(event_log):
        source = _find_video(root, item)
        if source is None:
            missing += 1
            continue

        category = _category(item)
        target = _slug(str(item.get("target_label", "unknown")))
        result = _slug(str(item.get("result_text", "unknown")))
        episode_number = int(item.get("episode_number", 0) or 0)
        destination_dir = view_root / category
        destination_dir.mkdir(exist_ok=True)
        destination = destination_dir / (
            f"{episode_number:05d}_goal-{target}_nav-{result}_{source.name}"
        )
        if destination.exists() or destination.is_symlink():
            if force:
                destination.unlink()
            else:
                counts[category] = counts.get(category, 0) + 1
                continue
        destination.symlink_to(source.resolve())
        counts[category] = counts.get(category, 0) + 1

    if missing:
        counts["missing_video"] = missing
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="为已有 ApexNav 安全评测视频创建攻击视角索引"
    )
    parser.add_argument("video_output_path", type=Path)
    parser.add_argument("--force", action="store_true", help="覆盖已有索引链接")
    args = parser.parse_args()

    counts = build_index(args.video_output_path, force=args.force)
    print(f"索引目录: {args.video_output_path / '_attack_view'}")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
