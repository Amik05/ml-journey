import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("1. imports...")
import numpy as np
from PIL import Image
import torch
print("2. basic imports ok")

from transformers import SamModel, SamProcessor
print("3. SAM imports ok")

from transformers import pipeline as hf_pipeline
print("4. pipeline import ok")

import open3d as o3d
print("5. open3d import ok")