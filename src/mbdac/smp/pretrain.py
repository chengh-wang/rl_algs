"""Standalone SMP epsilon-denoiser pretraining on native kinematic NPZ clips.

Every stride-one window belongs to one clip. All supplied references are training
samples; validation requires a separate, disjoint reference manifest (there is
no random split of overlapping windows). Quantiles are fitted on training only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .features import MotionFeatureBuffer, MotionFeatureSchema
from .vendor.model import DiffusionDenoiser
from .vendor.scheduler import DDPMScheduler



def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _references(source):
    source = Path(source).resolve()
    if source.is_dir():
        manifest = source / "manifest.json"
        if not manifest.exists():
            return {}, [(p, {}) for p in sorted(source.glob("*.npz"))]
        source = manifest
    metadata = json.loads(source.read_text())
    clips = metadata.get("clips", metadata.get("motions", []))
    records = []
    for clip in clips:
        info = {"path": clip} if isinstance(clip, str) else clip
        name = next((info[k] for k in ("path", "npz", "file", "npz_path", "output") if k in info), None)
        if name is None:
            raise ValueError("Each manifest clip needs a path to its NPZ")
        records.append(((source.parent / name).resolve(), info))
    return metadata, records


def build_motion_windows(references, window_size=10, window_batch_size=4096, *, schema=None):
    """Return CPU float32 [N,W,F], schema, and auditable source/window manifest."""
    if window_size <= 0 or window_batch_size <= 0:
        raise ValueError("Window and extraction batch sizes must be positive")
    metadata, records = _references(references)
    chunks, sources, windows = [], [], []
    if not isinstance(schema, MotionFeatureSchema):
        raise ValueError("An explicit MotionFeatureSchema is required; no robot layout is inferred")
    for path, info in records:
        sidecar = path.with_suffix(".json")
        local = {**metadata, **(json.loads(sidecar.read_text()) if sidecar.exists() else {}), **info}
        with np.load(path, allow_pickle=False) as data:
            def names(key):
                value = data[key].tolist() if key in data else local.get(key)
                if value is None:
                    raise ValueError(f"{path}: missing explicit {key}")
                result = tuple(str(x) for x in value)
                if len(set(result)) != len(result):
                    raise ValueError(f"{path}: duplicate {key}")
                return result
            joints, bodies = names("joint_names"), names("body_names")
            selected = schema.joint_names
            if any(n not in joints for n in selected) or any(n not in bodies for n in schema.end_effector_names):
                raise ValueError(f"{path}: schema contains unknown joint/end-effector names")
            fps = float(np.asarray(data["fps"]).item() if "fps" in data else local.get("fps", 1.0/schema.sample_dt))
            if not np.isfinite(fps) or fps <= 0 or not np.isclose(1.0/fps, schema.sample_dt):
                raise ValueError(f"{path}: sample interval differs from explicit schema")
            joint_ids = [joints.index(n) for n in selected]
            ee_ids = [bodies.index(n) for n in schema.end_effector_names]
            root_id = bodies.index(schema.root_body_name)
            def array(key, tail, body_key=None):
                raw = data[key] if key in data else data[body_key][:, root_id]
                value = torch.from_numpy(np.array(raw, dtype=np.float32, copy=True))
                if value.ndim != len(tail) + 1 or tuple(value.shape[1:]) != tail or not torch.isfinite(value).all():
                    raise ValueError(f"{path}: invalid {key} shape or nonfinite values")
                return value
            frames = dict(
                root_pos_w=array("root_pos_w", (3,), "body_pos_w"),
                root_quat_w=array("root_quat_w", (4,), "body_quat_w"),
                root_lin_vel_w=array("root_lin_vel_w", (3,), "body_lin_vel_w"),
                root_ang_vel_w=array("root_ang_vel_w", (3,), "body_ang_vel_w"),
                joint_pos=array("joint_pos", (len(joints),))[:, joint_ids],
                joint_vel=array("joint_vel", (len(joints),))[:, joint_ids],
                ee_pos_w=array("body_pos_w", (len(bodies), 3))[:, ee_ids],
            )
            count = len(frames["root_pos_w"])
            if any(len(v) != count for v in frames.values()) or count < window_size:
                raise ValueError(f"{path}: inconsistent lengths or clip shorter than window")
            if not torch.allclose(frames["root_quat_w"].norm(dim=-1), torch.ones(count), atol=1e-3):
                raise ValueError(f"{path}: root quaternions must be unit wxyz")
            source_index = len(sources)
            for offset in range(0, count - window_size + 1, window_batch_size):
                starts = torch.arange(offset, min(offset + window_batch_size, count - window_size + 1))
                indices = starts[:, None] + torch.arange(window_size)[None, :]
                buffer = MotionFeatureBuffer(len(starts), window_size, len(selected), len(ee_ids), "cpu")
                for key, value in frames.items():
                    getattr(buffer, key).copy_(value[indices])
                chunks.append(buffer.compute_features())
                windows.extend({"clip": source_index, "start_frame": int(start), "stop_frame": int(start) + window_size}
                               for start in starts)
            sources.append({"path": str(path), "sha256": _hash(path), "frame_count": count,
                            "window_count": count - window_size + 1, "metadata": {k: v for k, v in local.items() if k not in ("clips", "motions", "windows")}})
    if not chunks:
        raise ValueError("No reference clips found")
    result = torch.cat(chunks)
    if not torch.isfinite(result).all():
        raise ValueError("Feature extraction produced nonfinite values")
    return result, schema, {"clips": sources, "windows": windows, "nclips": len(sources),
                            "nwindows": len(result), "split": "train_only", "window_stride": 1}


def fit_quantiles(windows, span_floor=1e-4):
    if not np.isfinite(span_floor) or span_floor <= 0:
        raise ValueError("span_floor must be finite and positive")
    low, high = torch.quantile(windows.reshape(-1, windows.shape[-1]), torch.tensor([.01, .99]), dim=0)
    return low, torch.maximum(high, low + span_floor)


def diffusion_loss(model, scheduler, x_0, num_noise_samples=10):
    """Upstream _diffusion_loss: independent timestep/noise draws in B,K order."""
    if num_noise_samples <= 0:
        raise ValueError("num_noise_samples must be positive")
    batch_size = x_0.shape[0]
    expanded = x_0[:, None].expand(batch_size, num_noise_samples, *x_0.shape[1:]).reshape(
        batch_size * num_noise_samples, *x_0.shape[1:])
    timestep = scheduler.sample_timesteps(batch_size * num_noise_samples, x_0.device)
    noise = torch.randn_like(expanded)
    noisy = scheduler.add_noise(expanded, noise, timestep)
    return torch.nn.functional.l1_loss(model(noisy, timestep), noise)


def main(argv=None, *, schema_override=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", required=True, help="Native NPZ directory or JSON manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-schema", required=schema_override is None, help="Explicit named MotionFeatureSchema JSON")
    parser.add_argument("--validation-references", help="Separate disjoint clips; never a random window split")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--window-batch-size", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--span-floor", type=float, default=1e-4)
    parser.add_argument("--num-noise-samples", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.num_timesteps, args.num_layers, args.d_model, args.nhead, args.num_noise_samples, args.save_every, args.log_every) <= 0:
        parser.error("Epochs, batch size, and model dimensions must be positive")
    if args.learning_rate <= 0 or not np.isfinite(args.learning_rate):
        parser.error("Learning rate must be finite and positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Weight decay must be finite and nonnegative")
    if not np.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0:
        parser.error("Gradient norm limit must be finite and positive")
    if args.d_model % 2 or args.d_model % args.nhead:
        parser.error("d-model must be even and divisible by nhead")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output)
    if (output / "prior.pt").exists():
        raise FileExistsError(f"Refusing to overwrite an existing prior: {output / 'prior.pt'}; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    if schema_override is not None:
        if not isinstance(schema_override, MotionFeatureSchema):
            raise ValueError("schema_override must be a MotionFeatureSchema")
        if args.feature_schema is not None:
            raise ValueError("Choose schema_override or --feature-schema, not both")
        schema = schema_override
    else:
        schema_values = json.loads(Path(args.feature_schema).read_text())
        for key in ("joint_names", "end_effector_names"):
            schema_values[key] = tuple(schema_values[key])
        schema = MotionFeatureSchema(**schema_values)
    train, schema, manifest = build_motion_windows(args.references, args.window_size, args.window_batch_size, schema=schema)
    low, high = fit_quantiles(train, args.span_floor)
    validation = None
    if args.validation_references:
        validation, val_schema, val_manifest = build_motion_windows(args.validation_references, args.window_size, args.window_batch_size, schema=schema)
        if val_schema != schema:
            raise ValueError("Validation schema differs from training")
        if {c["sha256"] for c in manifest["clips"]} & {c["sha256"] for c in val_manifest["clips"]}:
            raise ValueError("Validation and training contain identical clips")
        manifest["split"] = "explicit_clip_holdout"
        manifest["validation"] = val_manifest
    feature_schema = asdict(schema)
    cfg = {**vars(args), "feature_dim": schema.feature_dim, "dropout": 0.0,
           "loss": "epsilon_l1", "use_ema": False}
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (output / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2) + "\n")
    (output / "window_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    np.savez_compressed(output / "windows.npz", features=train.numpy(), q_low=low.numpy(), q_high=high.numpy())
    report = {"nclips": manifest["nclips"], "nwindows": len(train), "shape": list(train.shape),
              "validation_windows": 0 if validation is None else len(validation),
              "split": manifest["split"], "normalization": "training q01/q99; span floor; no clipping",
              "span_floor": args.span_floor, "cfg": cfg, "epochs": []}
    def write_report():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_report()
    if args.prepare_only:
        print(json.dumps(report))
        return
    device = torch.device(args.device)
    model = DiffusionDenoiser(cfg["feature_dim"], args.window_size, args.d_model, args.nhead, args.num_layers).to(device)
    scheduler = DDPMScheduler(args.num_timesteps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    normalized = 2 * (train - low) / (high - low + 1e-8) - 1
    loader = DataLoader(TensorDataset(normalized), batch_size=args.batch_size, shuffle=True)
    val_loader = None if validation is None else DataLoader(TensorDataset(2 * (validation-low)/(high-low+1e-8)-1), batch_size=args.batch_size)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = diffusion_loss(model, scheduler, batch, args.num_noise_samples)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(batch)
        metrics = {"epoch": epoch, "train_l1": total / len(train)}
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    val_total += diffusion_loss(model, scheduler, batch, args.num_noise_samples).item() * len(batch)
            metrics["validation_l1"] = val_total / len(validation)
        report["epochs"].append(metrics)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint = dict(cfg=cfg, model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                              optimizer=optimizer.state_dict(), torch_rng_state=torch.get_rng_state(),
                              cuda_rng_states=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                              q_low=low, q_high=high, feature_schema=feature_schema, epoch=epoch,
                              reference_manifest=manifest, normalization=report["normalization"], span_floor=args.span_floor,
                              nclips=manifest["nclips"], nwindows=len(train), metrics=metrics)
            temporary = output / "prior.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "prior.pt")
        if epoch % args.log_every == 0 or epoch == args.epochs:
            write_report()
            print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
