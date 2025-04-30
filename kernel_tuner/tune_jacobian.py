import kernel_tuner
import numpy as np
import math
import os
import argparse
from common import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"

def main():
    parser = argparse.ArgumentParser(description="Tune Jacobian kernel.")
    parser.add_argument("--cache", type=str, help="The cache name", default=None)
    parser.add_argument("--strategy", type=str, help="The strategy name", default=None)
    cli_args = parser.parse_args()

    N, K = 224, 5
    # N, K = 255, 1
    nvoxels = np.int32(N * N)
    nreadouts = np.int32(N * K)
    nsamples_per_readout = np.int32(N)
    ncoils = np.int32(4)

    Jv = random_complex(ncoils, nreadouts, nsamples_per_readout)
    echos = random_complex(nreadouts, nvoxels)
    delta_echos_T1 = random_complex(nreadouts, nvoxels)
    delta_echos_T2 = random_complex(nreadouts, nvoxels)
    parameters = random_complex(10, nvoxels)
    coil_sensitivities = random_complex(ncoils, nvoxels)
    E = random_complex(nsamples_per_readout, nvoxels)
    dEdT2 = random_complex(nsamples_per_readout, nvoxels)
    v = random_complex(4, nvoxels)

    args = [
        nvoxels,
        nreadouts,
        nsamples_per_readout,
        ncoils,
        Jv,
        echos,
        delta_echos_T1,
        delta_echos_T2,
        nvoxels,
        parameters,
        coil_sensitivities,
        E,
        dEdT2,
        v
    ]

    template_args = ",".join([
        "THREADS_PER_ITEM",
        "SAMPLES_PER_THREAD",
        "READOUTS_PER_THREAD",
        "COILS_PER_THREAD",
        "BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z",
        "BLOCKS_PER_SM"
    ])

    kernel_name = f"compas::kernels::jacobian_product<{template_args}>"
    kernel_source = "#include \"jacobian/product_kernels.cuh\""
    lang = "cupy"
    compiler_options = [
        "-std=c++17",
        "-D__NVCC__=1",
        f"-I{ROOT_DIR}/kernel_tuner",
        f"-I{ROOT_DIR}/src",
        f"-I{ROOT_DIR}/include",
        f"-I{ROOT_DIR}/thirdparty/kmm/include",
    ]

    block_size_names = [f"BLOCK_SIZE_{c}" for c in "XYZ"]

    tune_params = dict()
    tune_params["BLOCK_SIZE_X"] = [16, 32, 64] #[1, 2, 4, 8, 16, 32, 64, 128]
    tune_params["BLOCK_SIZE_Y"] = [1, 2, 4, 8, 12, 16] #[1, 2, 4, 8, 16, 32, 64, 128]
    tune_params["BLOCK_SIZE_Z"] = [1]
    tune_params["THREADS_PER_ITEM"] = [1, 2, 4, 8, 16, 32]  # 1, 2, 4, 8, 16, 32]
    tune_params["COILS_PER_THREAD"] = [n for n in range(1, 9) if ncoils % n == 0]
    tune_params["SAMPLES_PER_THREAD"] = [n for n in range(1, 9) if nsamples_per_readout % n == 0]
    tune_params["READOUTS_PER_THREAD"] = [n for n in range(1, 9) if nreadouts % n == 0]
    tune_params["BLOCKS_PER_SM"] = [4, 8, 12, 16]

    restrictions = [
        "BLOCK_SIZE_X * BLOCK_SIZE_Y >= 64",
        "BLOCK_SIZE_X * BLOCK_SIZE_Y <= 1024",
        "BLOCK_SIZE_X >= THREADS_PER_ITEM",
        "SAMPLES_PER_THREAD * READOUTS_PER_THREAD <= 16",
        "BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * BLOCKS_PER_SM <= 2048",
    ]

    def problem_size(config):
        return [
            div_ceil(nsamples_per_readout * config["THREADS_PER_ITEM"], config["SAMPLES_PER_THREAD"]),
            div_ceil(nreadouts, config["READOUTS_PER_THREAD"]),
            div_ceil(ncoils, config["COILS_PER_THREAD"])
        ]

    kernel_tuner.tune_kernel(
        kernel_name,
        kernel_source,
        problem_size,
        args,
        tune_params,
        lang=lang,
        restrictions=restrictions,
        block_size_names=block_size_names,
        compiler_options=compiler_options,
        strategy=cli_args.strategy,
        cache=cli_args.cache,
    )


if __name__ == "__main__":
    main()
