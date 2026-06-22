#!/usr/bin/env python3
"""
ApexNav launcher utility.

Features:
- One-command start/stop/status/check for the full evaluation pipeline.
- Configurable run name, dataset, single-GPU or multi-GPU assignment.
- Optional service port offset and ROS master port for concurrent runs.
- Auto health check after startup to catch "fake online" failures early.

Usage examples:
  python tools/apexnav_launcher.py start --run-name hm3dv2_gpu5 --dataset hm3dv2 --gpus 5 --clean
  python tools/apexnav_launcher.py start --run-name hm3dv2_split --dataset hm3dv2 --gpus 4,5
  python tools/apexnav_launcher.py start --run-name hm3dv2_parallel_b --dataset hm3dv2 --gpus 6 --port-offset 100 --ros-master-port 11411
  python tools/apexnav_launcher.py status --run-name hm3dv2_gpu5 --dataset hm3dv2
  python tools/apexnav_launcher.py check --run-name hm3dv2_gpu5 --dataset hm3dv2
  python tools/apexnav_launcher.py stop --run-name hm3dv2_gpu5 --kill-processes
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]


def _default_conda_sh() -> str:
    candidates = [
        os.getenv("APEXNAV_CONDA_SH"),
        os.getenv("CONDA_SH"),
    ]
    conda_prefix = os.getenv("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
        candidates.extend(
            [
                str(prefix / "etc" / "profile.d" / "conda.sh"),
                str(prefix.parent.parent / "etc" / "profile.d" / "conda.sh"),
            ]
        )
    candidates.extend(
        [
            str(Path.home() / "anaconda3" / "etc" / "profile.d" / "conda.sh"),
            str(Path.home() / "miniconda3" / "etc" / "profile.d" / "conda.sh"),
            "/opt/conda/etc/profile.d/conda.sh",
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return "conda.sh"


DEFAULT_CONDA_SH = _default_conda_sh()
DEFAULT_APEXNAV_ENV = "apexnav"
DEFAULT_ROS_ENV = "apexnav311"
DEFAULT_ROS_MASTER_PORT = 11311
DEFAULT_PYTHON_EXEC = os.getenv("APEXNAV_PYTHON_EXEC", "python")
DEFAULT_QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
)
DEFAULT_QWEN_API_KEY = os.getenv(
    "QWEN_API_KEY", os.getenv("DEEPSEEK_API_KEY", "write your api key here")
)
DEFAULT_QWEN_MODEL = os.getenv(
    "QWEN_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
)

SERVICE_SPECS = [
    {
        "name": "gdino",
        "module": "vlm.detector.grounding_dino",
        "default_port": 12181,
        "env_var": "APEXNAV_GDINO_PORT",
    },
    {
        "name": "blip2",
        "module": "vlm.itm.blip2itm",
        "default_port": 12182,
        "env_var": "APEXNAV_BLIP2_PORT",
    },
    {
        "name": "sam",
        "module": "vlm.segmentor.sam",
        "default_port": 12183,
        "env_var": "APEXNAV_SAM_PORT",
    },
    {
        "name": "yolo",
        "module": "vlm.detector.yolov7",
        "default_port": 12184,
        "env_var": "APEXNAV_YOLO_PORT",
    },
]


def run(
    cmd: List[str], check: bool = True, capture: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
    )


def bash(cmd: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return run(["bash", "-lc", cmd], check=check, capture=capture)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def parse_gpu_list(raw: str) -> List[str]:
    gpus = [x.strip() for x in raw.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus 不能为空，例如 5 或 4,5")
    if any(not g.isdigit() for g in gpus):
        raise ValueError(f"GPU 列表格式错误: {raw}")
    return gpus


def bool_str(flag: bool) -> str:
    return "true" if flag else "false"


def q(s: str) -> str:
    return shlex.quote(s)


def ros_master_uri(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def tmux_has_session(session: str) -> bool:
    proc = run(["tmux", "has-session", "-t", session], check=False)
    return proc.returncode == 0


def tmux_list_windows(session: str) -> List[str]:
    proc = run(["tmux", "list-windows", "-t", session], check=False)
    if proc.returncode != 0:
        return []
    names = []
    for line in proc.stdout.splitlines():
        m = re.match(r"^\d+:\s+(.+?)\s+\(", line)
        if m:
            names.append(m.group(1).rstrip("*-").strip())
    return names


def tmux_window_exists(session: str, window_name: str) -> bool:
    return window_name in tmux_list_windows(session)


def tmux_ensure_session(session: str) -> None:
    if not tmux_has_session(session):
        run(["tmux", "new-session", "-d", "-s", session, "-n", "keepalive"], check=True)


def tmux_kill_window(session: str, window_name: str) -> None:
    if tmux_window_exists(session, window_name):
        run(["tmux", "kill-window", "-t", f"{session}:{window_name}"], check=False)


def tmux_new_window(session: str, window_name: str, command: str) -> None:
    run(["tmux", "new-window", "-t", session, "-n", window_name, command], check=True)


def service_specs_with_ports(port_offset: int) -> List[Dict[str, object]]:
    specs = []
    for spec in SERVICE_SPECS:
        port = int(spec["default_port"]) + port_offset
        if not 1 <= port <= 65535:
            raise ValueError(f"端口超出范围: {port}")
        specs.append(
            {
                "name": spec["name"],
                "module": spec["module"],
                "port": port,
                "env_var": spec["env_var"],
            }
        )
    return specs


def service_port_map(port_offset: int) -> Dict[str, int]:
    return {
        str(spec["name"]): int(spec["port"])
        for spec in service_specs_with_ports(port_offset)
    }


def runtime_env(args: argparse.Namespace) -> Dict[str, str]:
    env = {
        "ROS_MASTER_URI": ros_master_uri(args.ros_master_port),
        # The project uses local model files/caches; avoid slow HuggingFace HEAD
        # retries on restricted networks before each evaluation run.
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    for spec in service_specs_with_ports(args.port_offset):
        env[str(spec["env_var"])] = str(spec["port"])
    return env


def build_prefix(
    conda_sh: str,
    conda_env: str,
    gpu: Optional[str] = None,
    source_ros: bool = False,
    with_py39_shim: bool = False,
    extra_env: Optional[Mapping[str, str]] = None,
) -> str:
    cmds = [
        f"cd {q(str(ROOT))}",
        f"source {q(conda_sh)}",
        f"conda activate {q(conda_env)}",
    ]
    if source_ros:
        cmds.append("source devel/setup.bash")
    if with_py39_shim:
        cmds.append("unset PYTHONPATH")
        cmds.append(f"export PYTHONPATH={q(str(ROOT / 'ros_py39_shim'))}")
    if extra_env:
        for key, value in extra_env.items():
            cmds.append(f"export {key}={q(str(value))}")
    if gpu is not None:
        cmds.append(f"export CUDA_VISIBLE_DEVICES={q(gpu)}")
    return " && ".join(cmds)


def choose_service_gpus(gpus: List[str]) -> List[str]:
    return [gpus[i % len(gpus)] for i in range(len(SERVICE_SPECS))]


def eval_paths(run_name: str, dataset: str) -> Dict[str, str]:
    safe = sanitize_name(run_name)
    video_output_path = str(ROOT / "videos" / safe)
    llm_answer_path = str(ROOT / "llm" / "answers" / f"llm_answer_{dataset}_{safe}.txt")
    llm_response_path = str(
        ROOT / "llm" / "answers" / f"llm_response_{dataset}_{safe}.txt"
    )
    log_path = f"/tmp/apexnav_{safe}.log"
    return {
        "video_output_path": video_output_path,
        "llm_answer_path": llm_answer_path,
        "llm_response_path": llm_response_path,
        "log_path": log_path,
    }


def session_name(args: argparse.Namespace) -> str:
    return args.session or f"apexnav_{sanitize_name(args.run_name)}"


def _cleanup_patterns(
    args: argparse.Namespace,
    paths: Mapping[str, str],
    global_mode: bool,
    include_services: bool = True,
) -> List[str]:
    patterns = []
    if include_services:
        for spec in service_specs_with_ports(args.port_offset):
            port = int(spec["port"])
            module = str(spec["module"])
            patterns.append(f"pkill -f {q(f'{module} --port {port}')} || true")

    patterns.extend(
        [
            f"pkill -f {q(f'rosmaster --core -p {args.ros_master_port}')} || true",
            f"pkill -f {q(paths['video_output_path'])} || true",
            f"pkill -f {q(paths['llm_answer_path'])} || true",
            f"pkill -f {q(paths['log_path'])} || true",
        ]
    )

    if global_mode:
        patterns.extend(
            [
                "pkill -f 'habitat_evaluation.py --dataset' || true",
                "pkill -f 'roslaunch exploration_manager exploration.launch' || true",
            ]
        )

    return patterns


def cleanup_processes(
    args: argparse.Namespace,
    paths: Mapping[str, str],
    global_mode: bool = False,
    include_services: bool = True,
) -> None:
    patterns = _cleanup_patterns(
        args,
        paths,
        global_mode=global_mode,
        include_services=include_services,
    )
    if patterns:
        bash(" && ".join(patterns), check=False)


def _safe_head(path: Path, lines: int = 18) -> str:
    if not path.exists():
        return f"[MISSING] {path}"
    content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(content[:lines]) if content else "[EMPTY FILE]"


def _extract_metrics(path: Path) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    if not path.exists():
        return metrics

    content = path.read_text(encoding="utf-8", errors="ignore")
    for key in [
        "Average Success",
        "Average SPL",
        "Average Soft SPL",
        "Average Distance to Goal",
        "Total Success",
        "Total SPL",
        "Total Soft SPL",
        "Total Distance to Goal",
    ]:
        m = re.search(rf"\|\s*{re.escape(key)}\s*\|\s*([^|]+)\|", content)
        if m:
            metrics[key] = m.group(1).strip()

    m_no = re.search(r"No\.(\d+)\s+task is finished", content)
    if m_no:
        metrics["Latest Finished Task"] = m_no.group(1)

    return metrics


def _count_published_ports(ports: Iterable[int]) -> str:
    ports_pat = "|".join([f":{p}" for p in ports])
    proc = bash(f"ss -ltnp | grep -E '{ports_pat}' || true", check=False)
    return proc.stdout.strip()


def _service_ports_text(args: argparse.Namespace) -> str:
    ports = service_port_map(args.port_offset)
    ordered_names = ["gdino", "blip2", "sam", "yolo"]
    return ", ".join([f"{name}:{ports[name]}" for name in ordered_names])


def status_pipeline(args: argparse.Namespace) -> None:
    session = session_name(args)
    paths = eval_paths(args.run_name, args.dataset)
    ports = service_port_map(args.port_offset)

    print("=== ApexNav pipeline status ===")
    print(f"session: {session}")
    print(f"dataset: {args.dataset}")
    print(f"ros_master_uri: {ros_master_uri(args.ros_master_port)}")
    print(f"service_ports: {_service_ports_text(args)}")

    if tmux_has_session(session):
        print("\n[tmux windows]")
        proc = run(["tmux", "list-windows", "-t", session], check=False)
        print(proc.stdout.strip())
    else:
        print("\n[tmux windows]")
        print("session not found")

    print("\n[ports]")
    print(
        _count_published_ports(
            [args.ros_master_port, ports["gdino"], ports["blip2"], ports["sam"], ports["yolo"]]
        )
        or "no matching listening ports"
    )

    print("\n[continue.txt head]")
    continue_file = Path(paths["video_output_path"]) / "continue.txt"
    record_file = Path(paths["video_output_path"]) / "record.txt"
    print(_safe_head(continue_file))

    print("\n[record.txt head]")
    print(_safe_head(record_file))

    print("\n[key metrics]")
    metrics = _extract_metrics(record_file)
    if not metrics:
        metrics = _extract_metrics(continue_file)
    if metrics:
        for k, v in metrics.items():
            print(f"{k}: {v}")
    else:
        print("metrics not found yet")

    log_path = Path(paths["log_path"])
    print("\n[log tail]")
    if log_path.exists():
        proc = bash(f"tail -n 40 {q(str(log_path))} || true", check=False)
        print(proc.stdout.strip())
    else:
        print(f"[MISSING] {log_path}")


def check_pipeline(args: argparse.Namespace) -> None:
    if args.wait_seconds > 0:
        print(f"Waiting {args.wait_seconds}s before health check...")
        time.sleep(args.wait_seconds)

    status_pipeline(args)

    ros_prefix = build_prefix(
        conda_sh=args.conda_sh,
        conda_env=args.ros_env,
        source_ros=True,
        extra_env={"ROS_MASTER_URI": ros_master_uri(args.ros_master_port)},
    )

    print("\n[rostopic info /habitat/odom]")
    proc = bash(f"{ros_prefix} && rostopic info /habitat/odom || true", check=False)
    output = (proc.stdout or proc.stderr or "").strip()
    print(output or "no rostopic info output")

    print("\n[rostopic echo -n 1 /habitat/odom]")
    proc = bash(
        f"{ros_prefix} && timeout 10 rostopic echo -n 1 /habitat/odom | head -n 20 || true",
        check=False,
    )
    output = (proc.stdout or proc.stderr or "").strip()
    print(output or "no odom message observed within timeout")


def start_pipeline(args: argparse.Namespace) -> None:
    gpus = parse_gpu_list(args.gpus)
    service_gpus = choose_service_gpus(gpus)
    eval_gpu = args.eval_gpu if args.eval_gpu is not None else gpus[-1]

    if not Path(args.conda_sh).exists():
        raise FileNotFoundError(f"找不到 conda 初始化脚本: {args.conda_sh}")

    session = session_name(args)
    paths = eval_paths(args.run_name, args.dataset)
    env_map = runtime_env(args)

    if args.clean:
        if tmux_has_session(session):
            run(["tmux", "kill-session", "-t", session], check=False)
        cleanup_processes(
            args,
            paths,
            global_mode=False,
            include_services=not args.reuse_services,
        )

    if args.global_clean:
        cleanup_processes(
            args,
            paths,
            global_mode=True,
            include_services=not args.reuse_services,
        )

    tmux_ensure_session(session)

    if not args.reuse_services:
        for idx, spec in enumerate(service_specs_with_ports(args.port_offset)):
            win = f"{spec['name']}_{sanitize_name(args.run_name)}"
            if args.restart_window:
                tmux_kill_window(session, win)
            if tmux_window_exists(session, win):
                continue

            prefix = build_prefix(
                conda_sh=args.conda_sh,
                conda_env=args.apexnav_env,
                gpu=service_gpus[idx],
                source_ros=False,
                extra_env=env_map,
            )
            cmd = f"{prefix} && python -m {spec['module']} --port {spec['port']}"
            tmux_new_window(session, win, cmd)

    ros_win = f"ros_{sanitize_name(args.run_name)}"
    if args.restart_window:
        tmux_kill_window(session, ros_win)
    if not tmux_window_exists(session, ros_win):
        ros_prefix = build_prefix(
            conda_sh=args.conda_sh,
            conda_env=args.ros_env,
            source_ros=True,
            extra_env=env_map,
        )
        ros_cmd = f"{ros_prefix} && roslaunch exploration_manager exploration.launch"
        tmux_new_window(session, ros_win, ros_cmd)

    eval_win = f"eval_{sanitize_name(args.run_name)}"
    if args.restart_window:
        tmux_kill_window(session, eval_win)
    if not tmux_window_exists(session, eval_win):
        eval_env = dict(env_map)
        eval_env.update(
            {
                "QWEN_BASE_URL": args.qwen_base_url,
                "QWEN_API_KEY": args.qwen_api_key,
                "QWEN_MODEL": args.qwen_model,
            }
        )
        eval_prefix = build_prefix(
            conda_sh=args.conda_sh,
            conda_env=args.apexnav_env,
            gpu=str(eval_gpu),
            with_py39_shim=True,
            source_ros=False,
            extra_env=eval_env,
        )

        python_exec = q(args.python_exec)
        run_eval = (
            f"stdbuf -oL -eL {python_exec} habitat_evaluation.py "
            f"--dataset {q(args.dataset)} "
            f"need_video={bool_str(args.need_video)} "
            f"video_output_path={q(paths['video_output_path'])} "
            f"llm.llm_answer_path={q(paths['llm_answer_path'])} "
            f"llm.llm_response_path={q(paths['llm_response_path'])}"
        )

        if args.test_epi_num >= 0:
            run_eval += f" test_epi_num={int(args.test_epi_num)}"

        if args.hydra_override:
            run_eval += " " + " ".join([q(item) for item in args.hydra_override])

        full_eval_cmd = (
            f"{eval_prefix} && {run_eval} 2>&1 | tee -a {q(paths['log_path'])}"
        )
        tmux_new_window(session, eval_win, full_eval_cmd)

    print("=== ApexNav pipeline started ===")
    print(f"session: {session}")
    print(f"run_name: {args.run_name}")
    print(f"dataset: {args.dataset}")
    if args.reuse_services:
        print("service_gpus: reused_existing_services")
    else:
        print(f"service_gpus: {','.join(service_gpus)}")
    print(f"eval_gpu: {eval_gpu}")
    print(f"ros_master_uri: {ros_master_uri(args.ros_master_port)}")
    print(f"service_ports: {_service_ports_text(args)}")
    print(f"video_output_path: {paths['video_output_path']}")
    print(f"continue_file: {Path(paths['video_output_path']) / 'continue.txt'}")
    print(f"record_file: {Path(paths['video_output_path']) / 'record.txt'}")
    print(f"log_file: {paths['log_path']}")

    if args.check_after_seconds > 0:
        print(f"\nWaiting {args.check_after_seconds}s before automatic health check...")
        time.sleep(args.check_after_seconds)
        check_args = argparse.Namespace(**vars(args))
        check_args.wait_seconds = 0
        print()
        check_pipeline(check_args)


def stop_pipeline(args: argparse.Namespace) -> None:
    session = session_name(args)
    paths = eval_paths(args.run_name, args.dataset)

    if tmux_has_session(session):
        run(["tmux", "kill-session", "-t", session], check=False)

    if args.kill_processes:
        cleanup_processes(
            args,
            paths,
            global_mode=False,
            include_services=not args.keep_services,
        )

    if args.global_kill:
        cleanup_processes(
            args,
            paths,
            global_mode=True,
            include_services=not args.keep_services,
        )

    print("=== ApexNav pipeline stopped ===")
    print(f"session: {session}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ApexNav one-file launcher")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--run-name",
        type=str,
        default="hm3dv2_gpu5_video",
        help="评测名称，用于输出目录和文件名",
    )
    common.add_argument(
        "--dataset",
        type=str,
        choices=["hm3dv1", "hm3dv2", "mp3d"],
        default="hm3dv2",
        help="仿真数据集环境",
    )
    common.add_argument(
        "--session",
        type=str,
        default="",
        help="tmux session 名称，默认 apexnav_<run-name>",
    )
    common.add_argument("--ros-master-port", type=int, default=DEFAULT_ROS_MASTER_PORT)
    common.add_argument(
        "--port-offset",
        type=int,
        default=0,
        help="给 4 个 VLM 服务端口统一加偏移量；并行多 run 时很有用",
    )
    common.add_argument("--conda-sh", type=str, default=DEFAULT_CONDA_SH)
    common.add_argument("--ros-env", type=str, default=DEFAULT_ROS_ENV)

    start = sub.add_parser("start", parents=[common], help="启动整套流程")
    start.add_argument("--gpus", type=str, default="5", help="GPU 列表，单卡: 5，多卡: 4,5")
    start.add_argument(
        "--eval-gpu",
        type=str,
        default=None,
        help="可选：指定 eval 使用的 GPU，默认使用 --gpus 最后一个",
    )
    start.add_argument("--need-video", action="store_true", default=True, help="是否导出视频（默认开启）")
    start.add_argument("--no-video", dest="need_video", action="store_false", help="关闭视频导出")
    start.add_argument(
        "--test-epi-num",
        type=int,
        default=-1,
        help="单集调试用 episode id，默认 -1 表示跑全量",
    )
    start.add_argument(
        "--hydra-override",
        nargs="*",
        default=[],
        help="额外 Hydra 覆盖项，例如 habitat.environment.max_episode_steps=300",
    )
    start.add_argument("--restart-window", action="store_true", help="若窗口已存在则重建窗口")
    start.add_argument(
        "--reuse-services",
        action="store_true",
        help="复用已常驻的 4 个感知服务端口，不新建/重启 gdino、blip2、sam、yolo 窗口",
    )
    start.add_argument("--clean", action="store_true", help="启动前清理当前 run 相关 session 和端口进程")
    start.add_argument(
        "--global-clean",
        action="store_true",
        help="按你原始手工命令做更激进的全局 pkill，可能影响其他 run",
    )
    start.add_argument(
        "--check-after-seconds",
        type=int,
        default=30,
        help="启动后等待多少秒自动健康检查；设为 0 可关闭",
    )

    start.add_argument("--apexnav-env", type=str, default=DEFAULT_APEXNAV_ENV)
    start.add_argument(
        "--python-exec",
        type=str,
        default=DEFAULT_PYTHON_EXEC,
    )
    start.add_argument("--qwen-base-url", type=str, default=DEFAULT_QWEN_BASE_URL)
    start.add_argument("--qwen-api-key", type=str, default=DEFAULT_QWEN_API_KEY)
    start.add_argument(
        "--qwen-model",
        type=str,
        default=DEFAULT_QWEN_MODEL,
    )

    stop = sub.add_parser("stop", parents=[common], help="停止整套流程")
    stop.add_argument("--kill-processes", action="store_true", help="额外清理当前 run 相关端口和输出绑定进程")
    stop.add_argument(
        "--keep-services",
        action="store_true",
        help="配合 --kill-processes/--global-kill 使用时保留 4 个长期感知服务",
    )
    stop.add_argument(
        "--global-kill",
        action="store_true",
        help="额外执行全局 pkill，可能影响其他 run",
    )

    sub.add_parser("status", parents=[common], help="查看运行状态与核心指标文件")

    check = sub.add_parser("check", parents=[common], help="执行健康检查（窗口、端口、odom、指标）")
    check.add_argument("--wait-seconds", type=int, default=0, help="检查前等待多少秒")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.cmd == "start":
            start_pipeline(args)
        elif args.cmd == "stop":
            stop_pipeline(args)
        elif args.cmd == "status":
            status_pipeline(args)
        elif args.cmd == "check":
            check_pipeline(args)
        else:
            parser.error(f"未知命令: {args.cmd}")
            return 2
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
