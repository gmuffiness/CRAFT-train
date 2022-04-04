import os
import sys
from functools import wraps
import logging
import psutil
import time

def memory_printer(function):
    # general RAM usage
    @wraps(function)
    def wrapped(*args, **kwargs):
        # AFTER  code
        memory_usage_dict = dict(psutil.virtual_memory()._asdict())
        memory_usage_percent = memory_usage_dict['percent']
        print(f"Function name : {function.__name__}")
        print(f"BEFORE CODE: memory_usage_percent: {memory_usage_percent}%")
        # current process RAM usage
        pid = os.getpid()
        current_process = psutil.Process(pid)
        current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2. ** 20
        print(f"BEFORE CODE: Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")
        print("--" * 30)
        result = function(*args, **kwargs)
        memory_usage_dict = dict(psutil.virtual_memory()._asdict())
        memory_usage_percent = memory_usage_dict['percent']
        print(f"AFTER CODE: memory_usage_percent: {memory_usage_percent}%")
        # current process RAM usage
        pid = os.getpid()
        current_process = psutil.Process(pid)
        current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2. ** 20
        print(f"AFTER CODE: Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")
        print("--" * 30)
        return result
    return wrapped

def time_printer(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        # logging.info(f"{function.__qualname__}")
        start_time = time.time()
        result = function(*args, **kwargs)
        # logging.info(f"function {function.__qualname__} took {time.time() - start_time}")
        print(f"Function {function.__name__} took {round(time.time() - start_time, 2)} sec")
        return result
    return wrapped