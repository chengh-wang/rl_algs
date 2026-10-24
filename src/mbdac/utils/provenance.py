"""Small, non-secret runtime manifest for reproducible comparisons."""
import importlib.metadata
import platform
import subprocess
import torch


def runtime_provenance():
    result={'python':platform.python_version(),'platform':platform.platform(),'packages':{}}
    for name in ['mbdac','torch','tensordict','numpy','mjlab','mujoco','mujoco-warp','warp-lang','rsl-rl-lib','wandb']:
        try: result['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    try:
        result['git_revision']=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
        result['git_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    except (OSError,subprocess.CalledProcessError):
        result['git_revision']=None
    result['torch_cuda']=torch.version.cuda
    if torch.cuda.is_available(): result['gpu']=torch.cuda.get_device_name(0)
    return result
