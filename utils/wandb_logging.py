import torch.nn as nn
from typing import Optional, Union
from pathlib import Path
 
try:
 
    import wandb
    
    _WANDB_AVAILABLE = True
 
except ImportError:
 
    _WANDB_AVAILABLE = False
 
 
class WandbLogger:
 
    def __init__(self,project:str,run_name:Optional[str] = None,config:Optional[dict] = None,mode:str = "online",enabled:bool = True):
 
        if mode not in ("online", "offline", "disabled"):
 
            raise ValueError("mode must be 'online', 'offline', or 'disabled'.")
 
        self.enabled = enabled and _WANDB_AVAILABLE
        self.run = None
 
        if enabled and not _WANDB_AVAILABLE:
 
            print("wandb is not installed, logging is disabled. Run `pip install wandb` to enable it.")
 
        if self.enabled:
 
            self.run = wandb.init(project=project,name=run_name,config=config,mode=mode)
 
    def log(self,metrics:dict,step:Optional[int] = None):
 
        if not self.enabled:
 
            return
 
        wandb.log(metrics,step=step)
 
    def watch(self,model:nn.Module,log:str = "gradients",log_freq:int = 100):
 
        if not self.enabled:
 
            return
 
        wandb.watch(model,log=log,log_freq=log_freq)
 
    def log_checkpoint(self,checkpoint_path:Union[str, Path],name:str = "model"):
 
        if not self.enabled:
 
            return
 
        artifact = wandb.Artifact(name=name,type="model")
        artifact.add_file(str(checkpoint_path))
        self.run.log_artifact(artifact)
 
    def finish(self):
 
        if not self.enabled:
 
            return
 
        wandb.finish()