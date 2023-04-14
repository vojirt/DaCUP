import sys
import os
from pathlib import Path
import ipdb
import traceback
import torch 

class Logger(object):
    def __init__(self, out_dir="./", filename="console.log"):
        os.makedirs(out_dir, exist_ok=True)
        self.terminal = sys.stdout
        self.log = (Path(out_dir) / filename).open(mode='a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        self.log.flush()

    def flush(self):
        self.log.flush()
        self.terminal.flush()

def with_debugger(orig_fn):
    def new_fn(*args, **kwargs):
        try:
            return orig_fn(*args, **kwargs)
        except Exception as e:
            if isinstance(sys.stdout, Logger):
                sys.stdout = sys.stdout.terminal
            print(traceback.format_exc())
            print(e)
            ipdb.post_mortem()

    return new_fn

def with_train_anomaly_detect(orig_fn):
    def new_fn(*args, **kwargs):
        with torch.autograd.detect_anomaly():
            return orig_fn(*args, **kwargs)
    return new_fn

