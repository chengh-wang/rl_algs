"""Publish completed native-velocity evidence without changing training history.

Dry-run is offline and uses only the standard library. Publishing reuses one
analysis run, preserves all online samples, and labels retrospective timings
separately. Training checkpoints, replay and trajectories are never uploaded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://api.forge.coreweave.com"
ENTITY = "sslab_motion_mimic-teams"
PROJECT = "mbdac-native-velocity"
TRIALS = {
    "native_go1_fastsac_v2": "Holosoma FastSAC",
    "native_go1_mbdac_v3": "MBDAC",
    "native_go1_wm_fastsac_v3": "WM + FastSAC",
    "native_go1_real_only_v3": "Same integration, real SAC only",
    "native_g1_wm_fastsac_v3": "G1 WM + FastSAC",
}
METRICS = ("xy_rmse_m_s", "yaw_rmse_rad_s", "survival", "mean_return")
EVAL_COLUMNS = ["trial", "label", "robot", "method", "recipe", "iteration",
                "real_transitions", "active_process_s", "process_elapsed_s",
                "iteration_logged_process_s", *METRICS, "pass", "all_command_pass"]
COMMAND_COLUMNS = ["trial", "label", "robot", "iteration", "active_process_s",
                   "process_elapsed_s", "iteration_logged_process_s", "real_transitions",
                   "command_vx_m_s", "command_vy_m_s", "command_yaw_rad_s", *METRICS]
SAFE_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt", ".png", ".mp4"}


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def relative(path):
    # Do not accidentally publish arbitrary external files through a symlink.
    return path.resolve().relative_to(ROOT).as_posix()


def collect(args):
    files, problems, trials = set(), [], []
    selected_path = args.results / "selected_policies.json"
    selected = read_json(selected_path, [])
    if not isinstance(selected, list):
        raise ValueError("selected_policies.json must contain a list")
    selected_names = {Path(row["run"]).name for row in selected}
    if selected_names != set(TRIALS) or len(selected) != len(TRIALS):
        problems.append("selected_policies.json must contain exactly the five completed trials")
    for row in selected:
        if not Path(row.get("video", "")).is_file():
            problems.append(f"Missing selected video: {row.get('video')}")
    for name, label in TRIALS.items():
        path = ROOT / "logs" / name
        timing = read_json(path / "timing.json", {})
        config = read_json(path / "config.json", {})
        identity = read_json(path / "wandb.json", {})
        rows = read_jsonl(path / "evaluations.jsonl")
        if timing.get("status") not in ("converged", "completed", "budget_reached"):
            problems.append(f"Trial is not completed normally: {name}")
        for filename in ("evaluations.jsonl", "timing.json", "config.json", "wandb.json"):
            if not (path / filename).is_file():
                problems.append(f"Missing training metadata: {name}/{filename}")
        if not identity.get("id") or not rows:
            problems.append(f"Missing W&B identity or online evaluations: {name}")
        metadata = [path / filename for filename in
                    ("evaluations.jsonl", "timing.json", "config.json", "runtime.json", "native_task.json")
                    if (path / filename).is_file()]
        files.update(metadata)
        trials.append(dict(name=name, label=label, path=path, identity=identity,
                           timing=timing, config=config, rows=rows, metadata=metadata))
    cost = read_json(args.refined / "offline_cost.json", {})
    sessions = cost.get("sessions", [])
    if not sessions or sessions[-1].get("status") != "completed":
        problems.append("Refined offline_cost.json must end with a completed cost session")
    for filename in ("checkpoint_evaluations.jsonl", "threshold_brackets.json", "protocol.json"):
        if not (args.refined / filename).is_file():
            problems.append(f"Missing completed retrospective output: {filename}")
    if not args.report.is_file():
        problems.append(f"Final report is missing: {args.report}")
    for directory in (args.results, args.refined):
        if directory.is_dir():
            files.update(path for path in directory.rglob("*")
                         if path.is_file() and path.suffix in SAFE_SUFFIXES
                         and path.name != "report_wandb.json"
                         and "wandb" not in path.relative_to(directory).parts)
    fixed = [args.report, ROOT / "doc/native_velocity_benchmark.md",
             ROOT / "requirements-native-velocity.lock.txt", ROOT / "README.md",
             ROOT / "pyproject.toml", ROOT / "LICENSE", Path(__file__).resolve(),
             ROOT / "scripts/setup_native_velocity_baseline.sh",
             ROOT / "logs/package_033_wheel_audit.json",
             ROOT / "dist/mbdac-0.3.3-py3-none-any.whl",
             ROOT / "dist/mbdac-0.3.3-source.tar.gz",
             ROOT / "dist/mbdac-0.3.3-source_manifest.json"]
    files.update(path for path in fixed if path.is_file())
    for path in fixed[-3:]:
        if not path.is_file():
            problems.append(f"Missing release artifact: {relative(path)}")
    files.update(path for path in (ROOT / "licenses").glob("*") if path.is_file())
    files.update((ROOT / "logs").glob("native*_*audit*.json"))
    # The sole .pt exception is a selected, standalone inference policy whose
    # export audit identifies the exact selected checkpoint. Never glob models.
    selected_checkpoints = {str(Path(row["checkpoint"]).resolve()) for row in selected}
    for audit_path in (ROOT / "logs/native_velocity_sdk_free_exports").glob("*/audit.json"):
        audit = read_json(audit_path, {})
        if str(Path(audit.get("checkpoint", "")).resolve()) in selected_checkpoints:
            files.add(audit_path)
            policy = audit_path.with_name("policy.pt")
            if policy.is_file():
                if digest(policy) != audit.get("export_sha256"):
                    problems.append(f"Export changed after its audit: {relative(policy)}")
                files.add(policy)
    files = sorted({path.resolve() for path in files if path.is_file()}, key=relative)
    inventory = [dict(path=relative(path), bytes=path.stat().st_size, sha256=digest(path)) for path in files]
    manifest_sha = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    return dict(files=files, inventory=inventory, manifest_sha256=manifest_sha,
                problems=problems, trials=trials, selected=selected, cost=cost,
                retrospective=read_jsonl(args.refined / "checkpoint_evaluations.jsonl"))


def table_rows(trials):
    evaluations, commands = [], []
    for trial in trials:
        for row in trial["rows"]:
            base = dict(row, trial=trial["name"], label=trial["label"],
                        recipe=trial["config"].get("recipe", "original"))
            evaluations.append([base.get(key) for key in EVAL_COLUMNS])
            for command in row.get("per_command", []):
                values = dict(base, **command)
                values.update(zip(("command_vx_m_s", "command_vy_m_s", "command_yaw_rad_s"), command["command"]))
                commands.append([values.get(key) for key in COMMAND_COLUMNS])
    return evaluations, commands


def charts(wandb, trials, prefix, axis, title):
    panels = (("mean_return", "Fixed physical evaluation return"),
              ("xy_rmse_m_s", "XY velocity RMSE (m/s; threshold 0.20)"),
              ("yaw_rmse_rad_s", "Yaw rate RMSE (rad/s; threshold 0.30)"))
    curves = []
    for trial in trials:
        rows = sorted((row for row in trial["rows"] if row.get(axis) is not None), key=lambda row: row[axis])
        if rows:
            curves.append((trial["label"], rows))
    return {f"{prefix}/{key}": wandb.plot.line_series(
        xs=[[row[axis] / 60 for row in rows] for _, rows in curves],
        ys=[[row[key] for row in rows] for _, rows in curves],
        keys=[label for label, _ in curves], title=f"{ylabel} | {title}", xname=title)
        for key, ylabel in panels} if curves else {}


def publish(args, evidence):
    if evidence["problems"]:
        raise ValueError("Publication is not ready:\n- " + "\n- ".join(evidence["problems"]))
    cache_path = args.results / "report_wandb.json"
    cache = read_json(cache_path, {})
    if cache and (cache.get("entity"), cache.get("project"), cache.get("base_url")) != (ENTITY, PROJECT, BASE_URL):
        raise ValueError("Cached report run belongs to another W&B destination")
    if cache.get("status") == "completed" and cache.get("manifest_sha256") == evidence["manifest_sha256"]:
        print(json.dumps(dict(status="unchanged", id=cache["id"], url=cache["url"])))
        return
    os.environ["WANDB_BASE_URL"] = BASE_URL
    import wandb
    if not cache:
        identity = uuid.uuid4().hex[:8]
        cache = dict(id=identity, entity=ENTITY, project=PROJECT, base_url=BASE_URL,
                     url=f"https://forge.coreweave.com/wandb/{ENTITY}/{PROJECT}/runs/{identity}")
    cache.update(status="publishing", manifest_sha256=evidence["manifest_sha256"])
    # Persist the id before network I/O, so an interrupted publish reuses it.
    cache_path.write_text(json.dumps(cache, indent=2) + "\n")
    run = wandb.init(entity=ENTITY, project=PROJECT, id=cache["id"], resume="allow",
                     name="Native velocity controlled comparison and physical evidence",
                     job_type="analysis", mode="online", dir=str(args.results),
                     settings=wandb.Settings(base_url=BASE_URL),
                     config=dict(purpose="Physical evaluation/report only; no training history edits",
                                 online_time="process_elapsed_s: original result availability",
                                 retrospective_time="iteration_logged_process_s: logged iteration estimate, not result availability"))
    try:
        cache["url"] = run.url
        run.config.update(dict(manifest_sha256=evidence["manifest_sha256"]), allow_val_change=True)
        online, commands = table_rows(evidence["trials"])
        payload = {"online/physical_evaluations": wandb.Table(columns=EVAL_COLUMNS, data=online),
                   "online/per_command": wandb.Table(columns=COMMAND_COLUMNS, data=commands)}
        go1 = [trial for trial in evidence["trials"] if trial["name"].startswith("native_go1_")]
        payload.update(charts(wandb, go1, "online_result_availability", "process_elapsed_s",
                              "Online result availability (wall minutes)"))
        payload.update(charts(wandb, go1, "online_active_process", "active_process_s",
                              "Scheduled online samples (active minutes; evaluation excluded)"))
        retrospective = []
        for trial in go1:
            retrospective.append(dict(trial, rows=[row for row in evidence["retrospective"]
                                                  if Path(row["run"]).name == trial["name"]]))
        if evidence["retrospective"]:
            rows, per_command = table_rows(retrospective)
            payload["retrospective/physical_evaluations"] = wandb.Table(columns=EVAL_COLUMNS, data=rows)
            payload["retrospective/per_command"] = wandb.Table(columns=COMMAND_COLUMNS, data=per_command)
            payload.update(charts(wandb, retrospective, "retrospective_iteration_time", "iteration_logged_process_s",
                                  "Retrospective logged iteration time (minutes; NOT result availability)"))
        sessions = evidence["cost"].get("sessions", [])
        cost_keys = ["status", "started_utc", "waiting_s", "offline_post_wait_elapsed_s", "newly_completed_result_rows", "note"]
        payload["retrospective/cost_sessions"] = wandb.Table(columns=cost_keys,
                                                           data=[[row.get(k) for k in cost_keys] for row in sessions])
        for path in evidence["files"]:
            key = "media/" + relative(path).replace(".", "_")
            if path.suffix == ".png":
                payload[key] = wandb.Image(str(path), caption=relative(path))
            elif path.suffix == ".mp4":
                payload[key] = wandb.Video(str(path), format="mp4", caption=relative(path))
        run.log(payload)
        artifact = wandb.Artifact("native-velocity-benchmark-v3", type="evaluation-report",
                                  metadata=dict(manifest_sha256=evidence["manifest_sha256"],
                                                training_checkpoints_included=False))
        for path in evidence["files"]:
            artifact.add_file(str(path), name=relative(path))
        run.log_artifact(artifact, aliases=["latest"])
        # Fill strict/per-command evidence as files on the ended training runs;
        # uploading files does not resume a learner or modify scalar history.
        api = wandb.Api(overrides={"base_url": BASE_URL}, timeout=60)
        for trial in evidence["trials"]:
            remote = api.run(f"{ENTITY}/{PROJECT}/{trial['identity']['id']}")
            if remote.state not in ("finished", "failed", "crashed"):
                raise ValueError(f"Refusing to attach files to a live training run: {trial['name']}")
            for path in trial["metadata"]:
                remote.upload_file(str(path), root=str(ROOT))
        run.summary.update(dict(manifest_sha256=evidence["manifest_sha256"],
                                file_count=len(evidence["files"]), online_evaluations=len(online),
                                per_command_evaluations=len(commands),
                                offline_post_wait_s=evidence["cost"].get("total_offline_post_wait_s"),
                                report_file=relative(args.report),
                                caveat="One-seed nominal pilot; samples joined by lines do not establish monotonic learning or an exact threshold crossing."))
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise
    cache.update(status="completed", published_utc=datetime.now(timezone.utc).isoformat())
    cache_path.write_text(json.dumps(cache, indent=2) + "\n")
    print(json.dumps(dict(status="completed", id=cache["id"], url=cache["url"], files=len(evidence["files"]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "logs/native_velocity_v3_final")
    parser.add_argument("--refined", type=Path, default=ROOT / "logs/native_go1_refined_checkpoints")
    parser.add_argument("--report", type=Path, default=ROOT / "doc/native_velocity_results.md")
    parser.add_argument("--dry-run", action="store_true", help="Offline inventory; never initialize W&B or create a run")
    args = parser.parse_args()
    args.results, args.refined, args.report = args.results.resolve(), args.refined.resolve(), args.report.resolve()
    for path in (args.results, args.refined, args.report):
        relative(path)
    evidence = collect(args)
    if args.dry_run:
        rows, commands = table_rows(evidence["trials"])
        print(json.dumps(dict(dry_run=True, ready=not evidence["problems"],
                              blocking_requirements=evidence["problems"],
                              file_count=len(evidence["files"]),
                              total_bytes=sum(row["bytes"] for row in evidence["inventory"]),
                              manifest_sha256=evidence["manifest_sha256"],
                              online_evaluation_rows=len(rows), per_command_rows=len(commands),
                              retrospective_rows=len(evidence["retrospective"]),
                              cache_file=str(args.results / "report_wandb.json"),
                              inventory=evidence["inventory"]), indent=2))
    else:
        publish(args, evidence)


if __name__ == "__main__":
    main()
