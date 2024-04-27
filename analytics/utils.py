import os
import shutil
from functools import wraps
from time import time

def cleanup(directory):
    
    for root, dirs, files in os.walk(directory):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
        

def display_runtime(func):
    @wraps(func)
    def timer(*args, **kwargs):
        start = time()
        out = func(*args, **kwargs)
        end = time()

        print(f"{func.__name__}: {format(end-start, '.2f')}s")
        return out
    return timer
