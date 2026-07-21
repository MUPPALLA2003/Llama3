import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from pathlib import Path
from typing import Optional,Union

class CheckpointManager:
  
    def __init__(self,checkpoint_dir:Union[str, Path],keep_last_k:int = 3,mode: str = "min"):

        if mode not in ("min", "max"):

            raise ValueError("mode must be 'min' or 'max'.")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = keep_last_k
        self.mode = mode
        self.best_metric = None
        self.best_path = self.checkpoint_dir / "best.pt"

    def _is_better(self,metric: float) -> bool:

        if self.best_metric is None:

            return True

        if self.mode == "min":

            return metric < self.best_metric

        return metric > self.best_metric

    def save(self,model: nn.Module,optimizer: Optional[Optimizer] = None,scheduler: Optional[LRScheduler] = None,epoch: int = 0,step: int = 0,metric: Optional[float] = None,extra: Optional[dict] = None) -> Path:

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict()
                if optimizer is not None
                else None
            ),
            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
            "epoch": epoch,
            "step": step,
            "metric": metric,
            "extra": extra,
        }

        checkpoint_path = (self.checkpoint_dir/ f"checkpoint_epoch{epoch}_step{step}.pt")
        torch.save(checkpoint, checkpoint_path)
        self._remove_old_checkpoints()

        if metric is not None and self._is_better(metric):

            self.best_metric = metric
            torch.save(checkpoint, self.best_path)

        return checkpoint_path

    def _remove_old_checkpoints(self):

        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"),key=lambda x: x.stat().st_mtime)

        while len(checkpoints) > self.keep_last_k:

            checkpoints[0].unlink()
            checkpoints.pop(0)

    def load(self,checkpoint_path: Union[str, Path],model: nn.Module,optimizer: Optional[Optimizer] = None,scheduler: Optional[LRScheduler] = None,map_location: Optional[Union[str, torch.device]] = None):

        checkpoint = torch.load(checkpoint_path,map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])

        if (optimizer is not None and checkpoint["optimizer_state_dict"] is not None):

            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if (scheduler is not None and checkpoint["scheduler_state_dict"] is not None):

            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint

    def load_latest(self,model: nn.Module,optimizer: Optional[Optimizer] = None,scheduler: Optional[LRScheduler] = None,map_location: Optional[Union[str, torch.device]] = None):

        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"),key=lambda x: x.stat().st_mtime)

        if len(checkpoints) == 0:

            return None

        return self.load(checkpoints[-1],model,optimizer,scheduler,map_location)

    def load_best(self,model: nn.Module,optimizer: Optional[Optimizer] = None,scheduler: Optional[LRScheduler] = None,map_location: Optional[Union[str, torch.device]] = None):

        if not self.best_path.exists():

            return None

        return self.load(self.best_path,model,optimizer,scheduler,map_location)