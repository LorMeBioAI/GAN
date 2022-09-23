# Implementation of matplotlib function
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.colors import LogNorm

image=Image.open("prepare/sum/00_001.jpg",mode='r')
print(image)