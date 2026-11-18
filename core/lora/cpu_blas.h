/* Optional LP64 OpenBLAS for adapter preparation. Each compose worker calls it with one
 * BLAS thread. A missing or differently configured library leaves the scalar path available. */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <dlfcn.h>
#include <limits>
#include <new>
#include <utility>
#include <vector>

namespace rad::lora::cpu {

enum : int { row_major = 101, col_major = 102, no_trans = 111, trans = 112, upper = 121 };

struct Blas {
    using Gemm = void (*)(int, int, int, int, int, int, float, const float*, int,
                          const float*, int, float, float*, int);
    using Syrk = void (*)(int, int, int, int, int, double, const double*, int, double, double*, int);
    using Geqrf = void (*)(int*, int*, float*, int*, float*, float*, int*, int*);
    using Orgqr = void (*)(int*, int*, int*, float*, int*, float*, float*, int*, int*);
    // OpenBLAS LAPACK exposes the Fortran interface, including trailing CHARACTER lengths.
    using Gesvd = void (*)(char*, char*, int*, int*, float*, int*, float*, float*, int*, float*,
                           int*, float*, int*, int*, size_t, size_t);
    Gemm gemm = nullptr;
    Syrk syrk = nullptr;
    Geqrf geqrf = nullptr;
    Orgqr orgqr = nullptr;
    Gesvd gesvd = nullptr;
};

template<class Fn> Fn symbol(void* library, const char* name) {
    return reinterpret_cast<Fn>(dlsym(library, name));
}

inline const Blas* blas() {
    struct Library {
        void* handle = dlopen("libopenblas.so.0", RTLD_NOW | RTLD_LOCAL);
        Blas api;
        bool ready = false;
        Library() {
            if (!handle) return;
            const auto config = symbol<const char* (*)()>(handle, "openblas_get_config");
            const auto parallel = symbol<int (*)()>(handle, "openblas_get_parallel");
            const auto threads = symbol<int (*)()>(handle, "openblas_get_num_threads");
            // A serial OpenBLAS build may omit locking around its shared buffers.
            if (!config || !parallel || !threads || !config() ||
                std::strstr(config(), "USE64BITINT") || !parallel() || threads() != 1) return;
            api.gemm = symbol<decltype(api.gemm)>(handle, "cblas_sgemm");
            api.syrk = symbol<decltype(api.syrk)>(handle, "cblas_dsyrk");
            api.geqrf = symbol<decltype(api.geqrf)>(handle, "sgeqrf_");
            api.orgqr = symbol<decltype(api.orgqr)>(handle, "sorgqr_");
            api.gesvd = symbol<decltype(api.gesvd)>(handle, "sgesvd_");
            ready = api.gemm && api.syrk && api.geqrf && api.orgqr && api.gesvd;
        }
        ~Library() { if (handle) dlclose(handle); }
    };
    static const Library library;
    return library.ready ? &library.api : nullptr;
}

/* Optional workspaces can fail to allocate even when dimensions fit the guard. */
template<class Work> bool try_accelerated(Work&& work) {
    try {
        return work();
    } catch (const std::bad_alloc&) {
        return false;
    }
}

/* Bound a worker's extra allocation and every BLAS dimension. Large shapes can use the
 * existing path. Multiplication is written this way so untrusted dimensions cannot overflow. */
inline bool fits(int64_t rows, int64_t cols, int64_t elements) {
    constexpr int64_t max_int = std::numeric_limits<int>::max();
    return rows > 0 && cols > 0 && rows <= max_int && cols <= max_int && rows <= elements / cols;
}

inline bool energy(const Blas& api, const std::vector<float>& B, const std::vector<float>& A,
                   int64_t rows, int64_t rank, int64_t cols, double* result) {
    constexpr int64_t max_elements = 4 * 1024 * 1024;
    if (rank < 8 || rank > 1024 || !fits(rows, rank, max_elements) ||
        !fits(cols, rank, max_elements)) return false;
    std::vector<double> bb((size_t)(rank * rank)), aa((size_t)(rank * rank));
    {
        const std::vector<double> values(B.begin(), B.end());
        api.syrk(row_major, upper, trans, rank, rows, 1.0, values.data(), rank, 0.0,
                 bb.data(), rank);
    }
    {
        const std::vector<double> values(A.begin(), A.end());
        api.syrk(row_major, upper, no_trans, rank, cols, 1.0, values.data(), cols, 0.0,
                 aa.data(), rank);
    }
    double value = 0.0;
    for (int64_t i = 0; i < rank; ++i)
        for (int64_t j = i; j < rank; ++j)
            value += (i == j ? 1.0 : 2.0) * bb[i * rank + j] * aa[i * rank + j];
    if (!std::isfinite(value) || value < 0.0) return false;
    *result = value;
    return true;
}

inline bool workspace(float optimal, std::vector<float>* scratch, int* count) {
    constexpr int max_elements = 4 * 1024 * 1024;
    if (!std::isfinite(optimal) || optimal < 1 || optimal > max_elements) return false;
    *count = (int)std::ceil(optimal);
    scratch->resize((size_t)*count);
    return true;
}

/* Replace a column-major matrix with thin Q and copy out its triangular R. */
inline bool thin_qr(const Blas& api, std::vector<float>* matrix, int rows, int cols,
                    std::vector<float>* triangle) {
    std::vector<float> tau((size_t)cols), scratch;
    float optimal = 0.0f;
    int count = -1, info = 0;
    api.geqrf(&rows, &cols, matrix->data(), &rows, tau.data(), &optimal, &count, &info);
    if (info || !workspace(optimal, &scratch, &count)) return false;
    api.geqrf(&rows, &cols, matrix->data(), &rows, tau.data(), scratch.data(), &count, &info);
    if (info) return false;
    triangle->assign((size_t)cols * cols, 0.0f);
    for (int j = 0; j < cols; ++j)
        std::copy_n(matrix->data() + (size_t)j * rows, j + 1, triangle->data() + (size_t)j * cols);
    count = -1;
    api.orgqr(&rows, &cols, &cols, matrix->data(), &rows, tau.data(), &optimal, &count, &info);
    if (info || !workspace(optimal, &scratch, &count)) return false;
    api.orgqr(&rows, &cols, &cols, matrix->data(), &rows, tau.data(), scratch.data(), &count, &info);
    return !info;
}

inline bool can_compress(int64_t rows, int64_t cols, int64_t rank, int64_t keep) {
    constexpr int64_t max_elements = 4 * 1024 * 1024;
    return keep > 0 && keep < rank && rank <= 1024 && rank <= std::min(rows, cols) &&
           fits(rows, rank, max_elements) && fits(cols, rank, max_elements);
}

inline bool compress(const Blas& api, const std::vector<float>& B, const std::vector<float>& A,
                     int rows, int cols, int rank, int keep, std::vector<float>* out_b,
                     std::vector<float>* out_a, double* retained) {
    // B = Qb Rb and A^T = Qa Ra. Only Rb Ra^T needs an SVD.
    std::vector<float> left((size_t)rows * rank), right(A);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < rank; ++j) left[i + (size_t)j * rows] = B[(size_t)i * rank + j];
    std::vector<float> rl, rr;
    if (!thin_qr(api, &left, rows, rank, &rl) || !thin_qr(api, &right, cols, rank, &rr)) return false;
    const size_t square = (size_t)rank * rank;
    std::vector<float> core(square), u(square), vt(square), values((size_t)rank), scratch;
    api.gemm(col_major, no_trans, trans, rank, rank, rank, 1, rl.data(), rank,
             rr.data(), rank, 0, core.data(), rank);
    char all = 'A';
    int count = -1, info = 0;
    float optimal = 0;
    api.gesvd(&all, &all, &rank, &rank, core.data(), &rank, values.data(), u.data(), &rank,
              vt.data(), &rank, &optimal, &count, &info, 1, 1);
    if (info || !workspace(optimal, &scratch, &count)) return false;
    api.gesvd(&all, &all, &rank, &rank, core.data(), &rank, values.data(), u.data(), &rank,
              vt.data(), &rank, scratch.data(), &count, &info, 1, 1);
    if (info) return false;
    double total = 0.0, top = 0.0;
    for (int j = 0; j < rank; ++j) {
        if (!std::isfinite(values[j])) return false;
        const double squared = (double)values[j] * values[j];
        total += squared;
        if (j >= keep) continue;
        top += squared;
        for (int i = 0; i < rank; ++i) u[i + (size_t)j * rank] *= values[j];
    }
    std::vector<float> b((size_t)rows * keep), a((size_t)cols * keep);
    api.gemm(col_major, trans, trans, keep, rows, rank, 1, u.data(), rank,
             left.data(), rows, 0, b.data(), keep);
    api.gemm(col_major, no_trans, trans, cols, keep, rank, 1, right.data(), cols,
             vt.data(), rank, 0, a.data(), cols);
    auto finite = [](float value) { return std::isfinite(value); };
    if (!std::all_of(b.begin(), b.end(), finite) || !std::all_of(a.begin(), a.end(), finite)) return false;
    *out_b = std::move(b);
    *out_a = std::move(a);
    *retained = total > 0.0 ? top / total : 1.0;
    return true;
}

/* Apply w1 ⊗ w2 to all sketch columns in one group. Pack the sketch so W2's inputs are
 * contiguous and reuse one multiplication across columns. Keep W2's own factors when present. */
inline bool apply_lokr(const Blas& api, bool transpose, const float* input, float* output,
                       int64_t columns, int64_t a, int64_t b, int64_t c, int64_t e, int64_t r2,
                       const float* w1, const float* w2, const float* w2a, const float* w2b) {
    constexpr int64_t max_elements = 4 * 1024 * 1024;
    // Check both flattened shapes as well as the matrices given to GEMM.
    if (!fits(a, c, max_elements) || !fits(b, e, max_elements) || !fits(b, c, max_elements) ||
        !fits(a * c, columns, max_elements) || !fits(b * e, columns, max_elements) ||
        !fits(b * c, columns, max_elements) ||
        (!w2 && !fits(b * columns, r2, max_elements))) return false;

    const int64_t batch = b * columns;
    std::vector<float> input_tiles((size_t)(batch * (transpose ? c : e)));
    std::vector<float> output_tiles((size_t)(batch * (transpose ? e : c)));
    std::vector<float> core((size_t)(batch * c));
    auto pack = [&](const float* from, int64_t width) {
        for (int64_t j = 0; j < b; ++j)
            for (int64_t h = 0; h < width; ++h)
                for (int64_t q = 0; q < columns; ++q)
                    input_tiles[(j * columns + q) * width + h] = from[(j * width + h) * columns + q];
    };
    auto unpack = [&](float* to, int64_t width) {
        for (int64_t j = 0; j < b; ++j)
            for (int64_t h = 0; h < width; ++h)
                for (int64_t q = 0; q < columns; ++q)
                    to[(j * width + h) * columns + q] = output_tiles[(j * columns + q) * width + h];
    };
    if (transpose) {
        api.gemm(row_major, trans, no_trans, b, c * columns, a, 1, w1, b,
                 input, c * columns, 0, core.data(), c * columns);
        pack(core.data(), c);
    } else {
        pack(input, e);
    }

    const int64_t in_width = transpose ? c : e;
    const int64_t out_width = transpose ? e : c;
    const int transform = transpose ? no_trans : trans;
    if (w2) {
        api.gemm(row_major, no_trans, transform, batch, out_width, in_width, 1,
                 input_tiles.data(), in_width, w2, e, 0, output_tiles.data(), out_width);
    } else {
        std::vector<float> intermediate((size_t)(batch * r2));
        api.gemm(row_major, no_trans, transform, batch, r2, in_width, 1, input_tiles.data(),
                 in_width, transpose ? w2a : w2b, transpose ? r2 : e, 0, intermediate.data(), r2);
        api.gemm(row_major, no_trans, transform, batch, out_width, r2, 1, intermediate.data(), r2,
                 transpose ? w2b : w2a, transpose ? e : r2, 0, output_tiles.data(), out_width);
    }
    if (transpose) {
        unpack(output, e);
    } else {
        unpack(core.data(), c);
        api.gemm(row_major, no_trans, no_trans, a, c * columns, b, 1, w1, b,
                 core.data(), c * columns, 0, output, c * columns);
    }
    return true;
}

}  // namespace rad::lora::cpu
