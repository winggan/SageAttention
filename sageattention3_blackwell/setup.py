import os
import sys
from pathlib import Path
from packaging.version import parse, Version
from setuptools import setup, find_packages
import subprocess
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "sageattn3"

# CUTLASS is consumed via the git submodule at csrc/cutlass, pinned at the
# tag below (see .gitmodules for the version record; the exact commit is
# recorded by git in the submodule gitlink).
# The tag is only used for diagnostics below; the build itself never runs git.
CUTLASS_TAG = "v4.3.4"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FAHOPPER_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = (
    os.getenv("FAHOPPER_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
    or "sdist" in sys.argv
)
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FAHOPPER_FORCE_CXX11_ABI", "FALSE") == "TRUE"



def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]


cmdclass = {}
ext_modules = []

if not SKIP_CUDA_BUILD:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))

    if CUDA_HOME is None:
        raise RuntimeError(
            "Cannot find CUDA_HOME. CUDA must be available to build the package.")

    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.8"):
        raise RuntimeError("Sage3 is only supported on CUDA 12.8 and above")

    # The kernels only support the SM120 family (consumer Blackwell, RTX 50).
    # Always build all supported archs; sm_121a requires nvcc >= 12.9
    # (CUDA 12.8's nvcc rejects compute_121a).
    target_archs = ["120a"]
    if bare_metal_version >= Version("12.9"):
        target_archs.append("121a")
    print(f"Target architectures: {target_archs}")

    cc_flag = []
    for num in target_archs:
        cc_flag += ["-gencode", f"arch=compute_{num},code=sm_{num}"]

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    repo_dir = Path(this_dir)
    cutlass_dir = repo_dir / "csrc" / "cutlass"
    if not (cutlass_dir / "include" / "cutlass").exists():
        # CUTLASS is only consumed via the git submodule at csrc/cutlass
        # (pinned at v4.3.4, see .gitmodules). Never run git from here:
        # just detect a missing checkout and tell the user what to do.
        raise RuntimeError(
            f"CUTLASS {CUTLASS_TAG} not found at {cutlass_dir}. "
            "CUTLASS is provided by a git submodule that is not checked out. "
            "From the repository root, run:\n"
            "  git submodule update --init --recursive\n"
            "If you are building from an sdist (no git history), fetch the "
            "pinned version manually:\n"
            f"  git clone --depth 1 --branch {CUTLASS_TAG} "
            f"https://github.com/NVIDIA/cutlass.git {cutlass_dir}"
        )
    nvcc_flags = [
        "-O3",
        # "-O0",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        # "--ptxas-options=-v",  # printing out number of registers
        "--ptxas-options=--warn-on-local-memory-usage",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
        "-DQBLKSIZE=128",
        "-DKBLKSIZE=128",
        "-DCTA256",
        "-DDQINRMEM",
    ]
    include_dirs = [
        repo_dir / "sageattn3",
        cutlass_dir / "include",
        cutlass_dir / "tools" / "util" / "include",
    ]

    ext_modules.append(
        CUDAExtension(
            name="sageattn3.fp4attn_cuda",
            sources=[
                "sageattn3/blackwell/api.cu",
                "sageattn3/blackwell/cuda_driver_shim.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(
                    nvcc_flags + ["-DEXECMODE=0"] + cc_flag
                ),
            },
            include_dirs=include_dirs,
            # cuTensorMapEncodeTiled is interposed by cuda_driver_shim.cpp and
            # resolved from the driver at runtime, so there is no build-time
            # dependency on libcuda.
        )
    )
    ext_modules.append(
        CUDAExtension(
            name="sageattn3.fp4quant_cuda",
            sources=["sageattn3/quantization/fp4_quantization_4d.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(
                    nvcc_flags + ["-DEXECMODE=0"] + cc_flag
                ),
            },
            include_dirs=include_dirs,
        )
    )


class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        super().run()

setup(
    name=PACKAGE_NAME,
    version="1.0.0",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    description="FP4FlashAttention",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension}
    if ext_modules
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "triton",
        "einops",
        "packaging",
        "ninja",
    ],
)
