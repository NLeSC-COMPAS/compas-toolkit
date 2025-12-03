import kernel_tuner
import numpy as np
import math
import os
import argparse
from common import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"

def main():
    N, K = 244, 5
    parser = argparse.ArgumentParser(description="Tune the simulate_signal kernel.")
    parser.add_argument("--cache", type=str, help="The cache name", default=None)
    parser.add_argument("--strategy", type=str, help="The strategy name", default=None)
    parser.add_argument("--coils", type=int, help="The number of coils", default=1)
    parser.add_argument("--voxels", type=int, help="The number of voxels", default=N * N)
    parser.add_argument("--readouts", type=int, help="The number of readouts", default=N)
    parser.add_argument("--samples", type=int, help="The number of samples per readout", default=N * K)
    cli_args = parser.parse_args()

    nvoxels = np.int32(cli_args.voxels)
    nreadouts = np.int32(cli_args.readouts)
    nsamples_per_readout = np.int32(cli_args.samples)
    ncoils = np.int32(cli_args.coils)

    print("parameters:")
    print(" - voxels:", nvoxels)
    print(" - readouts:", nreadouts)
    print(" - samples per readout:", nsamples_per_readout)
    print(" - coils:", ncoils)
    print(" - cache:", cli_args.cache or "no")
    print("")

    signal = random_complex(ncoils, nreadouts, nsamples_per_readout)
    sample_decay = random_complex(nsamples_per_readout, nvoxels)
    readout_echos = random_complex(nreadouts, nvoxels)
    coil_sensitivities = random_complex(ncoils, nvoxels)

    args = [
            np.int32(0),
            np.int32(nvoxels),
            np.int32(ncoils),
            np.int32(nreadouts),
            np.int32(nsamples_per_readout),
            signal,
            sample_decay,
            readout_echos,
            coil_sensitivities
    ]

    template_args = ",".join([
        "BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z",
        "THREADS_PER_VOXEL",
        "SAMPLES_PER_THREAD",
        "READOUTS_PER_THREAD",
        "COILS_PER_THREAD",
        "BLOCKS_PER_SM"
    ])

    kernel_name = f"compas::kernels::sum_signal_cartesian<{template_args}>"
    kernel_source = "#include \"trajectories/signal_kernels.cuh\""
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
    tune_params["BLOCK_SIZE_X"] = [16, 32, 64]
    tune_params["BLOCK_SIZE_Y"] = [1, 2, 4, 8, 16]
    tune_params["BLOCK_SIZE_Z"] = [1, 2, 4]
    tune_params["THREADS_PER_VOXEL"] = [1, 2, 4, 8, 16, 32]
    tune_params["SAMPLES_PER_THREAD"] = [1, 2, 3, 4, 5, 6]
    tune_params["READOUTS_PER_THREAD"] = [1, 2, 3, 4, 5, 6]
    tune_params["COILS_PER_THREAD"] = [1, 2, 3, 4]
    tune_params["BLOCKS_PER_SM"] = [4, 8, 12, 16]

    restrictions = [
        "BLOCK_SIZE_X * BLOCK_SIZE_Y >= 64",
        "BLOCK_SIZE_X * BLOCK_SIZE_Y <= 1024",
        "BLOCK_SIZE_X >= THREADS_PER_VOXEL",
        "SAMPLES_PER_THREAD * READOUTS_PER_THREAD * COILS_PER_THREAD <= 32",
        "BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * BLOCKS_PER_SM <= 2048",
    ]

    def problem_size(config):
        return [
            div_ceil(nsamples_per_readout * config["THREADS_PER_VOXEL"], config["SAMPLES_PER_THREAD"]),
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
