from time import time
from typing import Any, Callable


def runtime(
    f: Callable[[], Any], model_name: str = ""
) -> tuple[float, tuple[float, float]]:
    start_time = time()
    res = f()
    end_time = time()
    execution_time = end_time - start_time
    if len(model_name) > 0:
        if execution_time > 60:
            print(f"Runtime for {model_name}: {execution_time / 60:.2f} minutes")
        else:
            print(f"Runtime for {model_name}: {execution_time:.2f} seconds")
    return execution_time, res
