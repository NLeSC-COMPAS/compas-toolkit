import os
import numpy as np
import math

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"

def div_ceil(p, q):
    return p // q + int(p % q != 0)


def random_float(*shape):
    return np.random.random(shape).astype(np.float32) * 2.0 - 1.0


def random_complex(*shape):
    theta = random_float(*shape) * 2.0 * math.pi
    mag = random_float(*shape)

    output = np.zeros((*shape, 2), dtype=np.float32)
    output[..., 0] = np.cos(theta) * mag
    output[..., 1] = np.sin(theta) * mag

    return output
