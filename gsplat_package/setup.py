import glob
import os
import os.path as osp
import platform
import sys

from setuptools import find_packages, setup

__version__ = "1.4.0"
# exec(open("gsplat/version.py", "r").read())

URL = "https://github.com/nerfstudio-project/gsplat"  # TODO

# 环境变量控制项
BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"

# 兼容常见开关：强制 CPU only / 禁用 CUDA 构建
FORCE_CUDA_ENV = os.getenv("FORCE_CUDA", None)  # "0" 表示强制禁用 CUDA
CPU_ONLY_FLAG = os.getenv("GSPLAT_CPU_ONLY", "0") == "1"

def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False

def _should_build_cuda():
    """决定是否构建 CUDA 扩展。"""
    if BUILD_NO_CUDA:
        return False
    if CPU_ONLY_FLAG:
        return False
    if FORCE_CUDA_ENV is not None and FORCE_CUDA_ENV.strip() == "0":
        return False
    # 如需更严格，也可在此检查 NVCC/CUDA 是否可用
    return True

def get_ext():
    """
    返回 cmdclass 的 build_ext。若 torch 不可用则返回空 dict，
    以避免在构建初期导入 torch 失败。
    """
    if not _should_build_cuda():
        return {}

    if not _torch_available():
        # Torch 不可用时，跳过扩展构建（而不是报错）
        print("[gsplat] Torch not found at build time. Skipping CUDA extension build.")
        return {}

    try:
        from torch.utils.cpp_extension import BuildExtension
        # 与原逻辑一致
        return {"build_ext": BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)}
    except Exception as e:
        print(f"[gsplat] Failed to import BuildExtension from torch: {e}")
        print("[gsplat] Skipping CUDA extension build.")
        return {}

def get_extensions():
    """
    返回需要构建的扩展列表。若不可构建则返回 []。
    """
    if not _should_build_cuda():
        print("[gsplat] CUDA build disabled by environment. No extensions will be built.")
        return []

    if not _torch_available():
        print("[gsplat] Torch not found at build time. Skipping CUDA extension build.")
        return []

    try:
        import torch
        from torch.__config__ import parallel_info
        from torch.utils.cpp_extension import CUDAExtension
    except Exception as e:
        print(f"[gsplat] Import error when preparing extensions: {e}")
        print("[gsplat] Skipping CUDA extension build.")
        return []

    extensions_dir = osp.join("gsplat", "cuda", "csrc")
    sources = glob.glob(osp.join(extensions_dir, "*.cu")) + glob.glob(
        osp.join(extensions_dir, "*.cpp")
    )

    # remove generated 'hip' files, in case of rebuilds
    sources = [path for path in sources if "hip" not in path]

    undef_macros = []
    define_macros = []

    if sys.platform == "win32":
        define_macros += [("gsplat_EXPORTS", None)]

    extra_compile_args = {"cxx": ["-O3"]}
    if os.name != "nt":  # Not on Windows:
        extra_compile_args["cxx"] += ["-Wno-sign-compare"]
    extra_link_args = [] if WITH_SYMBOLS else ["-s"]

    # OpenMP 配置
    try:
        info = parallel_info()
    except Exception:
        info = ""
    if (
        "backend: OpenMP" in info
        and "OpenMP not found" not in info
        and sys.platform != "darwin"
    ):
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_compile_args["cxx"] += ["/openmp"]
        else:
            extra_compile_args["cxx"] += ["-fopenmp"]
    else:
        print("[gsplat] Compiling without OpenMP...")

    # mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

    # NVCC flags
    nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
    nvcc_flags = [] if nvcc_flags_env == "" else nvcc_flags_env.split(" ")
    nvcc_flags += ["-O3", "--use_fast_math"]
    if LINE_INFO:
        nvcc_flags += ["-lineinfo"]

    # ROCm / CUDA 区分
    if getattr(torch.version, "hip", None):
        define_macros += [("USE_ROCM", None)]
        undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
    else:
        nvcc_flags += ["--expt-relaxed-constexpr"]

    if sys.platform == "win32":
        nvcc_flags += ["-DWIN32_LEAN_AND_MEAN"]

    extra_compile_args["nvcc"] = nvcc_flags

    extension = CUDAExtension(
        "gsplat.csrc",
        sources,
        include_dirs=[osp.join(extensions_dir, "third_party", "glm")],
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    return [extension]

# 根据是否应该构建扩展，准备参数
ext_modules = get_extensions()
cmdclass = get_ext()

setup(
    name="gsplat",
    version=__version__,
    description=" Python package for differentiable rasterization of gaussians",
    keywords="gaussian, splatting, cuda",
    url=URL,
    download_url=f"{URL}/archive/gsplat-{__version__}.tar.gz",
    python_requires=">=3.7",
    install_requires=[
        "jaxtyping",
        "rich>=12",
        # 保留 torch 依赖；但注意：若你使用 --no-deps 安装，此处不会自动装 torch
        "torch",
        "typing_extensions; python_version<'3.8'",
    ],
    extras_require={
        # dev dependencies. Install them by `pip install gsplat[dev]`
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.2",
            "pytest-xdist==2.5.0",
            "typeguard>=2.13.3",
            "pyyaml==6.0",
            "build",
            "twine",
            "ninja",
        ],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
    include_package_data=True,
)
