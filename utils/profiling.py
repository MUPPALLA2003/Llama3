import torch
import torch.distributed as dist
from pathlib import Path
from typing import Optional
from torch.profiler import profile,schedule,ProfilerActivity,tensorboard_trace_handler

class PyTorchProfiler:

    def __init__(self,log_dir:str = "logs/profiler",wait: int = 1,warmup: int = 1,active: int = 3,repeat: int = 1,accumulate_across_repeats: bool = True,export_mode: str = "chrome") -> None:

        if export_mode not in ("chrome", "tensorboard"):

            raise ValueError(
                f"export_mode must be 'chrome' or 'tensorboard', got {export_mode!r}. "
                "torch.profiler does not support registering an on_trace_ready "
                "handler AND calling export_chrome_trace() on the same profile "
                "instance, so this wrapper requires picking one per run."
            )
 
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.is_main_process = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
        self._wait = wait
        self._warmup = warmup
        self._active = active
        self._repeat = repeat
        self._accumulate_across_repeats = accumulate_across_repeats
        self._export_mode = export_mode
        self.activities = [ProfilerActivity.CPU]

        if torch.cuda.is_available():

            self.activities.append(ProfilerActivity.CUDA)
 
        self.profiler = None
 
    def _build_profiler(self) -> profile:

        trace_handler = None

        if self._export_mode == "tensorboard" and self.is_main_process:

            trace_handler = tensorboard_trace_handler(str(self.log_dir))
 
        return profile(
            activities=self.activities,
            schedule=schedule(
                wait=self._wait,
                warmup=self._warmup,
                active=self._active,
                repeat=self._repeat,
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_stack=False,
            acc_events=self._accumulate_across_repeats,
        )
 
    def start(self) -> None:
    
        self.profiler = self._build_profiler()
        self.profiler.start()
 
    def step(self) -> None:
       
        if self.profiler is None:

            raise RuntimeError(
                "step() called before start(). Use `with PyTorchProfiler(...) as p:` "
                "or call p.start() before stepping."
            )
        
        self.profiler.step()
 
    def stop(self) -> None:
    
        if self.profiler is None:

            raise RuntimeError("stop() called before start().")
        
        self.profiler.stop()
 
    def summary(self,sort_by: str = "self_cuda_time_total",row_limit: int = 20,print_table: bool = True) -> Optional[str]:
    
        if not self.is_main_process:

            return None
        
        if self.profiler is None:

            raise RuntimeError("summary() called before start()/stop().")
        
        if not torch.cuda.is_available():

            sort_by = "self_cpu_time_total"
 
        table = self.profiler.key_averages().table(sort_by=sort_by,row_limit=row_limit)

        if print_table:

            print(table)

        return table
 
    def export_chrome_trace(self,filename: str = "trace.json") -> None:
        
        if not self.is_main_process:

            return
        
        if self.profiler is None:

            raise RuntimeError("export_chrome_trace() called before start()/stop().")
        
        if self._export_mode != "chrome":

            raise RuntimeError(
                "export_chrome_trace() requires export_mode='chrome'. This "
                "instance was built with export_mode='tensorboard', so its "
                "trace was already consumed by the TensorBoard handler "
                "during stop(). Construct a separate PyTorchProfiler with "
                "export_mode='chrome' if you need both outputs."
            )
        
        self.profiler.export_chrome_trace(filename)
 
    def __enter__(self):

        self.start()

        return self
 
    def __exit__(self,exc_type,exc_val,exc_tb):

        self.stop()
       
        return False