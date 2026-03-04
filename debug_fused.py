import os
os.environ["TRITON_INTERPRET"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from test_local import test_numerical_correctness

try:
    test_numerical_correctness(7)
except Exception as e:
    import traceback
    traceback.print_exc()
