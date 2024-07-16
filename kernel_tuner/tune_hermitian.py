import kernel_tuner
import numpy as np
import math
import os
import argparse
from common import *

def main():
    parser = argparse.ArgumentParser(description="Tune Jacobian hermitian kernel.")
    parser.add_argument("--cache", type=str, help="The cache name", default=None)
    parser.add_argument("--strategy", type=str, help="The strategy name", default=None)
    parser.add_argument("--ncoils", type=int, help="The number of coils", default=2)
    cli_args = parser.parse_args()

    N, K = 224, 5
    # N, K = 255, 1
    nvoxels = np.int32(N * N)
    nreadouts = np.int32(N * K)
    nsamples_per_readout = np.int32(N)
    ncoils = np.int32(cli_args.ncoils)

    JHv = random_complex(4, nvoxels)
    echos = random_complex(nreadouts, nvoxels)
    delta_echos_T1 = random_complex(nreadouts, nvoxels)
    delta_echos_T2 = random_complex(nreadouts, nvoxels)
    parameters = random_complex(10, nvoxels)
    coil_sensitivities = random_complex(ncoils, nvoxels)
    E = random_complex(nsamples_per_readout, nvoxels)
    dEdT2 = random_complex(nsamples_per_readout, nvoxels)
    v = random_complex(ncoils, nreadouts, nvoxels)

    args = [
        nreadouts,
        nsamples_per_readout,
        nvoxels,
        ncoils,
        JHv,
        echos,
        delta_echos_T1,
        delta_echos_T2,
        nvoxels,
        parameters,
        coil_sensitivities,
        v,
        E,
        dEdT2
    ]

    template_args = ",".join([
        str(ncoils),
        "VOXEL_TILE_SIZE",
        "READOUT_TILE_SIZE",
        "SAMPLE_TILE_SIZE",
        "BLOCK_SIZE_X",
        "BLOCK_SIZE_Y",
        "BLOCK_SIZE_Z",
        "BLOCKS_PER_SM",
        "USE_SMEM",
    ])

    kernel_name = f"compas::kernels::jacobian_hermitian_product<{template_args}>"
    kernel_source = "#include \"jacobian/hermitian_kernels.cuh\""
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
    tune_params["BLOCKS_PER_SM"] = [1, 4, 8]
    tune_params["BLOCK_SIZE_X"] = [8, 16, 32, 64, 128]
    tune_params["BLOCK_SIZE_Y"] = [1, 2, 4, 8, 16]
    tune_params["BLOCK_SIZE_Z"] = [1, 2, 4, 8, 16]

    tune_params["READOUT_TILE_SIZE"] = [1, 2, 4, 8, 16, 32, 64]
    tune_params["SAMPLE_TILE_SIZE"] = [1, 2, 4, 8, 16, 32, 64]
    tune_params["VOXEL_TILE_SIZE"] = [16, 32, 64, 128]
    tune_params["USE_SMEM"] = [0, 1]

    restrictions = [
        "BLOCK_SIZE_X == VOXEL_TILE_SIZE",
        "BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z >= 64",
        "BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z <= 1024",
        "BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * BLOCKS_PER_SM <= 2048",
        #"READOUT_TILE_SIZE * SAMPLE_TILE_SIZE <= 64",
        "VOXEL_TILE_SIZE % BLOCK_SIZE_X == 0",
        "READOUT_TILE_SIZE % BLOCK_SIZE_Y == 0",
        "SAMPLE_TILE_SIZE % BLOCK_SIZE_Z == 0",
        f"{nreadouts} % READOUT_TILE_SIZE == 0",
        f"{nsamples_per_readout} % SAMPLE_TILE_SIZE == 0",
    ]

    problem_size = [nvoxels, 1, 1]

    results, env = kernel_tuner.tune_kernel(
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


    results = sorted(results, key=lambda p: p["time"])
    best_results = []

    print("finished tuning, best results:")

    for record in results:
        should_skip = False

        for r in best_results:
            if record["READOUT_TILE_SIZE"] % r["READOUT_TILE_SIZE"] == 0 and \
                    record["SAMPLE_TILE_SIZE"] % r["SAMPLE_TILE_SIZE"] == 0:
                should_skip = True

        if should_skip:
            continue

        best_results.append(record)
        print((
            record["SAMPLE_TILE_SIZE"],
            record["VOXEL_TILE_SIZE"],
            record["READOUT_TILE_SIZE"],
            record["BLOCK_SIZE_X"],
            record["BLOCK_SIZE_Y"],
            record["BLOCK_SIZE_Z"],
        ), record["time"])


if __name__ == "__main__":
    main()
