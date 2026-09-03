/*
 * Copyright (c) 2025 by SageAttention team.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Runtime resolution of the CUDA driver API used by the TMA kernels.
//
// The host-side TMA descriptor setup in CUTLASS/cute calls
// cuTensorMapEncodeTiled, which lives in libcuda.so.1. Linking the extension
// against libcuda at build time adds a hard NEEDED entry and makes the wheel
// fail to import on systems whose driver predates that symbol. Instead, this
// translation unit defines cuTensorMapEncodeTiled itself (interposition) and
// forwards to the real driver entry point resolved through the CUDA runtime
// API. This mirrors the sm90 handling in csrc/qattn/qk_int_sv_f8_cuda_sm90.cu.

#include <cuda.h>
#include <cuda_runtime_api.h>

struct _CudaDriverApiLoader {
    static bool initialized;

    static bool init() {
#define LOAD(name) name = load_api<name##_t>((const char *)#name)
        LOAD(cuTensorMapEncodeTiled);
        LOAD(cuDriverGetVersion);
#undef LOAD
        return true;
    }

#define DECL_API(name) \
    using name##_t = decltype(&::name); \
    static name##_t name
    DECL_API(cuTensorMapEncodeTiled);
    DECL_API(cuDriverGetVersion);
#undef DECL_API

    template <typename FuncType>
    static FuncType load_api(const char *name) {
        void *func = nullptr;
        cudaDriverEntryPointQueryResult qres = {};
        auto ret =
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12050)
            cudaGetDriverEntryPointByVersion(name, &func, CUDA_VERSION, cudaEnableDefault, &qres);
#else
            cudaGetDriverEntryPoint(name, &func, cudaEnableDefault, &qres);
#endif
        if (ret == cudaSuccess && qres == cudaDriverEntryPointSuccess && func != nullptr) {
            return reinterpret_cast<FuncType>(func);
        } else {
            return nullptr;
        }
    }

    static bool is_available() {
        return cuTensorMapEncodeTiled != nullptr;
    }
};

bool _CudaDriverApiLoader::initialized = _CudaDriverApiLoader::init();

bool is_available() {
    return _CudaDriverApiLoader::is_available();
}

#define DECL_API(name) _CudaDriverApiLoader::name##_t _CudaDriverApiLoader::name
DECL_API(cuTensorMapEncodeTiled);
DECL_API(cuDriverGetVersion);
#undef DECL_API

// Interposed driver entry point. The signature must stay identical to the
// declaration in cuda.h; any drift is a compile error, which keeps the
// interposition honest across CUDA upgrades.
extern "C" CUresult cuTensorMapEncodeTiled(
    CUtensorMap *tensorMap,
    CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank,
    void *globalAddress,
    const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides,
    const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides,
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
    auto real = _CudaDriverApiLoader::cuTensorMapEncodeTiled;
    if (real == nullptr) {
        return CUDA_ERROR_NOT_FOUND;
    }
    return real(tensorMap, tensorDataType, tensorRank, globalAddress, globalDim,
                globalStrides, boxDim, elementStrides, interleave, swizzle,
                l2Promotion, oobFill);
}

// cute's TMA copy traits query the driver version at runtime; forward that
// query as well so the extension never references libcuda directly.
extern "C" CUresult cuDriverGetVersion(int *driverVersion) {
    auto real = _CudaDriverApiLoader::cuDriverGetVersion;
    if (real == nullptr) {
        return CUDA_ERROR_NOT_FOUND;
    }
    return real(driverVersion);
}
