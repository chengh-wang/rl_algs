"""Retrospectively evaluate saved native-velocity checkpoints, without training.

Online evaluations are never modified. Logged iteration timestamps estimate when
an iteration was ready; they are NOT the original evaluation availability time.
Saved policies are discrete samples, so neither curves nor threshold brackets
imply monotonic learning or interpolate a policy at an unsaved transition count.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = re.compile(r"model_(\d+)\.pt$")
LABELS = {"mbdac": "MBDAC", "fastsac": "Holosoma FastSAC", "wm_fastsac": "WM + FastSAC"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def wait_for_finalizer(pid):
    """Avoid waiting on a reused PID, or on an already exited zombie."""
    while True:
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return
        if state == "Z" or not any(Path(arg.decode(errors="replace")).name == "finalize_native_velocity.py" for arg in command):
            return
        time.sleep(5)


def inventory(run):
    """Read counters/timestamps on CPU, without loading a policy or simulator."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    timing = json.loads((run / "timing.json").read_text())
    config = json.loads((run / "config.json").read_text())
    native_task = (run / "native_task.json").read_bytes().rstrip(b"\n")
    if hashlib.sha256(native_task).hexdigest() != timing["task_sha256"]:
        raise ValueError(f"Native task fingerprint does not match its manifest: {run}")
    events = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
    tag = "Loss/qf_loss" if timing["method"] == "fastsac" else "iteration"
    stamps = {}
    for event in events.Scalars(tag):
        if event.step in stamps and event.wall_time != stamps[event.step]:
            raise ValueError(f"Ambiguous repeated TensorBoard step {run}:{event.step}")
        stamps[event.step] = event.wall_time
    metrics = {int(row["iteration"]): row for row in read_jsonl(run / "metrics.jsonl")}
    checkpoints = []
    for path in run.glob("model_*.pt"):
        match = MODEL_NAME.fullmatch(path.name)
        if not match:
            continue
        iteration = int(match[1])
        if iteration not in stamps:
            raise ValueError(f"No exact {tag} timestamp for {path}")
        sidecar = path.with_suffix(".timing.json")
        saved_timing = json.loads(sidecar.read_text()) if sidecar.exists() else None
        if timing["method"] == "fastsac":
            if saved_timing is None or saved_timing["global_step"] != iteration:
                raise ValueError(f"Missing/mismatched exact source checkpoint counters: {path}")
            transitions = saved_timing["real_transitions"]
            counter_basis = str(sidecar)
        else:
            if iteration not in metrics:
                raise ValueError(f"No exact metrics row for {path}")
            transitions = metrics[iteration]["real_transitions"]
            counter_basis = f"{run / 'metrics.jsonl'}:iteration={iteration}"
        elapsed = stamps[iteration] - timing["process_start_unix"]
        if elapsed < 0:
            raise ValueError(f"Negative logged process time for {path}")
        checkpoints.append(dict(checkpoint=str(path), iteration=iteration,
            real_transitions=int(transitions), transition_counter_basis=counter_basis,
            iteration_logged_process_s=elapsed, iteration_log_tag=tag,
            iteration_log_wall_unix=stamps[iteration], source_save_timing=saved_timing))
    checkpoints.sort(key=lambda row: row["iteration"])
    if len({row["iteration"] for row in checkpoints}) != len(checkpoints):
        raise ValueError(f"Duplicate ordinary checkpoint iterations in {run}")
    initial = next((row for row in read_jsonl(run / "evaluations.jsonl") if row["iteration"] == 0), None)
    if initial is None or initial.get("all_command_pass", True):
        raise ValueError(f"Expected an existing failed initialization evaluation: {run}")
    return dict(run=str(run), timing=timing, recipe=config.get("recipe", "original"),
        config_sha256=sha256(run / "config.json"), initial=initial, checkpoints=checkpoints)


def protocol():
    from importlib.metadata import version

    files = ("examples/mjlab_velocity/evaluation.py", "examples/mjlab_velocity/evaluate.py",
             "examples/mjlab_velocity/environment.py", "examples/mjlab_velocity/config.py",
             "examples/mjlab_velocity/model_task.py", "examples/mjlab_velocity/kinematics.py",
             "src/mbdac/runners/model_based_runner.py", "src/mbdac/modules/networks.py",
             "src/mbdac/modules/holosoma.py", "src/mbdac/vendor/holosoma/fast_sac.py",
             "src/mbdac/vendor/holosoma/normalization.py")
    value = dict(seconds=20.0, replicas=8, seed=1701,
        evaluation_source_sha256={name: sha256(ROOT / name) for name in files},
        packages={name: version(name) for name in ("mbdac", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch")})
    return value, hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def append_jsonl(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
        stream.flush()


def plot(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (("mean_return", "Fixed evaluation return", None),
              ("xy_rmse_m_s", "XY velocity RMSE (m/s)", .20),
              ("yaw_rmse_rad_s", "Yaw rate RMSE (rad/s)", .30),
              ("survival", "20 s survival fraction", 1.0))
    for axis, scale, title in (("iteration_logged_process_s", 60., "Retrospective logged iteration time (minutes)"),
                               ("real_transitions", 1e6, "Real environment transitions (millions)")):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), layout="constrained")
        for run in dict.fromkeys(row["run"] for row in records):
            rows = [row for row in records if row["run"] == run and row.get(axis) is not None]
            if not rows:
                continue
            first = rows[0]
            label = "Same integration, real SAC only" if first["recipe"] == "real_only" else LABELS[first["method"]]
            for ax, (key, ylabel, threshold) in zip(axes.flat, panels):
                ax.plot([row[axis] / scale for row in rows], [row[key] for row in rows], marker="o", label=label)
                ax.set(xlabel=title, ylabel=ylabel)
                ax.grid(alpha=.25)
        for ax, (_, _, threshold) in zip(axes.flat, panels):
            if threshold is not None:
                ax.axhline(threshold, color=".4", linestyle="--", linewidth=1)
        axes[0, 0].legend()
        seeds = sorted({row["training_seed"] for row in records})
        fig.suptitle(f"Retrospective saved-checkpoint evaluation | {records[0]['robot'].upper()} | training seeds {seeds}\n"
                     "One-seed pilot; discrete samples, no interpolated threshold crossing")
        fig.savefig(output / f"retrospective_{axis}.png", dpi=180)
        plt.close(fig)


def run_evaluations(inventories, output, protocol_value, protocol_sha):
    # These imports can initialize CUDA. Call only after the finalizer has exited.
    sys.path.insert(0, str(ROOT))
    import torch
    from examples.mjlab_velocity.evaluate import load_policy
    from examples.mjlab_velocity.evaluation import evaluate

    torch.set_num_threads(4)
    result_path = output / "checkpoint_evaluations.jsonl"
    previous = read_jsonl(result_path)
    cached = {}
    configs = {item["run"]: item["config_sha256"] for item in inventories}
    for row in previous:
        path = Path(row["checkpoint"])
        if row["protocol_sha256"] != protocol_sha or not path.exists() or sha256(path) != row["checkpoint_sha256"]:
            raise ValueError(f"Cannot reuse changed checkpoint/evaluation protocol: {path}")
        if row["run"] not in configs or row["config_sha256"] != configs[row["run"]]:
            raise ValueError(f"Cannot reuse changed run configuration: {row['run']}")
        key = (row["run"], row["checkpoint"])
        if key in cached:
            raise ValueError(f"Duplicate completed result: {key}")
        cached[key] = row
    selected = []
    summaries = []
    for item in inventories:
        run = Path(item["run"]); timing = item["timing"]
        if timing.get("status") not in ("converged", "completed", "budget_reached"):
            raise ValueError(f"Training has not finalized: {run}")
        common = dict(run=str(run), method=timing["method"], robot=timing["robot"],
            training_seed=timing["seed"], recipe=item["recipe"], task_sha256=timing["task_sha256"],
            config_sha256=item["config_sha256"], protocol_sha256=protocol_sha)
        initial_path = run / "eval_model_0.pt"
        initial_key = (str(run), str(initial_path))
        if initial_key not in cached:
            initial = {**item["initial"], **common, "checkpoint": str(initial_path),
                "checkpoint_sha256": sha256(initial_path), "origin": "existing_online_initial_evaluation",
                "iteration_logged_process_s": None, "offline_checkpoint_work_s": 0.,
                "online_evaluation_file": str(run / "evaluations.jsonl")}
            append_jsonl(result_path, initial); cached[initial_key] = initial
        rows = [cached[initial_key]]
        for checkpoint in item["checkpoints"]:
            key = (str(run), checkpoint["checkpoint"])
            if key not in cached:
                began = time.perf_counter(); path = Path(checkpoint["checkpoint"])
                digest = sha256(path); policy, loaded = load_policy(path)
                if loaded["iteration"] != checkpoint["iteration"] or loaded["method"] != timing["method"]:
                    raise ValueError(f"Loaded checkpoint identity does not match metadata: {path}")
                loaded_s = time.perf_counter() - began
                metrics = evaluate(policy, timing["robot"], replicas=protocol_value["replicas"],
                    seconds=protocol_value["seconds"], seed=protocol_value["seed"])
                del policy
                row = {**metrics, **checkpoint, **common, "checkpoint_sha256": digest,
                    "origin": "offline_native_physical_evaluation", "checkpoint_hash_and_load_s": loaded_s,
                    "offline_checkpoint_work_s": time.perf_counter() - began,
                    "evaluated_utc": datetime.now(timezone.utc).isoformat()}
                append_jsonl(result_path, row); cached[key] = row
                print(json.dumps(dict(checkpoint=str(path), all_command_pass=row["all_command_pass"],
                    xy_rmse_m_s=row["xy_rmse_m_s"], yaw_rmse_rad_s=row["yaw_rmse_rad_s"],
                    real_transitions=row["real_transitions"])), flush=True)
            row = cached[key]
            if row["config_sha256"] != common["config_sha256"]:
                raise ValueError(f"Run config changed since evaluation: {run}")
            rows.append(row)
            if row["all_command_pass"]:
                break
        passing = rows[-1] if rows[-1]["all_command_pass"] else None
        summaries.append(dict(run=str(run), first_strict_sampled_pass=passing,
            preceding_sampled_failure=rows[-2] if passing and len(rows) > 1 else None,
            last_evaluated=rows[-1], ordinary_checkpoint_count_available=len(item["checkpoints"]),
            ordinary_checkpoint_count_evaluated=len(rows) - 1,
            bracket_caveat="Only adjacent evaluated checkpoints; no monotonicity or exact crossing time is assumed."))
        selected.extend(rows)
    (output / "threshold_brackets.json").write_text(json.dumps(summaries, indent=2) + "\n")
    plot(selected, output)
    return selected


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, help="Wait only while this PID is finalize_native_velocity.py")
    parser.add_argument("--metadata-only", action="store_true", help="Print CPU metadata inventory; no wait, files or GPU work")
    args = parser.parse_args(); runs = [path.resolve() for path in args.runs]
    if args.metadata_only:
        print(json.dumps([inventory(run) for run in runs], indent=2)); return
    if args.wait_pid:
        wait_for_finalizer(args.wait_pid)
    waiting_s = time.perf_counter() - started
    active_start = time.perf_counter(); output = args.out.resolve(); output.mkdir(parents=True, exist_ok=True)
    status = "error"; before = len(read_jsonl(output / "checkpoint_evaluations.jsonl"))
    try:
        inventories = [inventory(run) for run in runs]
        if len({(x["timing"]["task_sha256"], x["timing"]["robot"], x["timing"]["seed"]) for x in inventories}) != 1:
            raise ValueError("Common plots require matching native task, robot and training seed")
        value, digest = protocol()
        protocol_path = output / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text())["protocol_sha256"] != digest:
            raise ValueError("Existing results use another evaluation protocol; choose a fresh output directory")
        protocol_path.write_text(json.dumps(dict(protocol=value, protocol_sha256=digest,
            time_basis="TensorBoard wall_time minus original process start: retrospective logged iteration estimate, NOT original result availability.",
            transition_basis="Exact matching metrics row or source save sidecar; no iteration-derived counters or interpolation."), indent=2) + "\n")
        run_evaluations(inventories, output, value, digest); status = "completed"
    finally:
        cost = dict(status=status, started_utc=datetime.fromtimestamp(time.time() - (time.perf_counter() - started), timezone.utc).isoformat(),
            waiting_s=waiting_s, offline_post_wait_elapsed_s=time.perf_counter() - active_start,
            script_pre_exit_elapsed_s=time.perf_counter() - started,
            newly_completed_result_rows=len(read_jsonl(output / "checkpoint_evaluations.jsonl")) - before,
            note="Separate post-hoc cost; never added to original checkpoint training timestamps. Includes parsing, hashing, loading, evaluating and plotting; excludes time waiting for the finalizer.")
        append_jsonl(output / "offline_cost_sessions.jsonl", cost)
        sessions = read_jsonl(output / "offline_cost_sessions.jsonl")
        (output / "offline_cost.json").write_text(json.dumps(dict(sessions=sessions,
            total_offline_post_wait_s=sum(row["offline_post_wait_elapsed_s"] for row in sessions)), indent=2) + "\n")


if __name__ == "__main__":
    main()
