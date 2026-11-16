//-------------------------------------------------------------------------------------
// OptimizeLRUDegenerateTest.cpp
//
// Regression test: OptimizeFacesLRU / OptimizeFacesLRUEx must be robust on
// degenerate meshes (zero-area triangles, isolated vertices, duplicate faces,
// and faces referencing the "unused" index value).
//
// For every input this test asserts:
//   1. The call succeeds and never reads/writes out of bounds (run under
//      ASan/UBSan or a checked build to catch memory errors).
//   2. The output face remap contains only valid face indices or UNUSED32,
//      and every face without an unused index appears exactly once.
//   3. The per-face vertex order is untouched (the remap only reorders faces).
//   4. The optimized order's FIFO vertex-cache hit rate is at least as good
//      as plain sequential traversal.
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//-------------------------------------------------------------------------------------

#include <DirectXMesh.h>

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace DirectX;

namespace
{
    int g_failures = 0;

    // Matches the library-internal DirectX::UNUSED32 sentinel.
    constexpr uint32_t kUnusedFace = 0xFFFFFFFFu;

    void ReportFailure(const char* caseName, const char* what)
    {
        std::printf("FAIL %-24s %s\n", caseName, what);
        ++g_failures;
    }

    template<typename IndexT>
    constexpr IndexT kUnused = IndexT(-1);

    template<typename IndexT>
    bool IsValidFace(const IndexT* indices, uint32_t face)
    {
        return indices[size_t(face) * 3] != kUnused<IndexT>
            && indices[size_t(face) * 3 + 1] != kUnused<IndexT>
            && indices[size_t(face) * 3 + 2] != kUnused<IndexT>;
    }

    // FIFO vertex-cache miss count over a sequence of faces, matching the
    // model used by the OptimizeFacesLRU implementation.
    template<typename IndexT>
    uint64_t SimulateFifoMisses(const IndexT* indices, const std::vector<uint32_t>& faceOrder, uint32_t cacheSize)
    {
        std::vector<IndexT> fifo;
        fifo.reserve(cacheSize);
        uint32_t head = 0;
        uint64_t misses = 0;

        for (uint32_t face : faceOrder)
        {
            for (uint32_t j = 0; j < 3; ++j)
            {
                const IndexT v = indices[size_t(face) * 3 + j];

                bool hit = false;
                for (IndexT c : fifo)
                {
                    if (c == v)
                    {
                        hit = true;
                        break;
                    }
                }
                if (hit)
                    continue;

                ++misses;
                if (fifo.size() < cacheSize)
                {
                    fifo.push_back(v);
                }
                else
                {
                    fifo[head] = v;
                    head = (head + 1) % cacheSize;
                }
            }
        }

        return misses;
    }

    template<typename IndexT>
    void RunCase(const char* name, const std::vector<IndexT>& indices, size_t nVerts, uint32_t lruCacheSize)
    {
        const size_t nFaces = indices.size() / 3;
        std::vector<uint32_t> remap(nFaces, 0xDEADBEEFu);
        const std::vector<IndexT> indicesBefore(indices);

        const HRESULT hr = OptimizeFacesLRU(indices.data(), nFaces, nVerts, remap.data(), lruCacheSize);
        if (FAILED(hr))
        {
            ReportFailure(name, "OptimizeFacesLRU failed");
            return;
        }

        // (3) Vertex order semantics: the call must not touch the input
        //     indices, and emitted triangles are the input triangles (same
        //     winding), merely reordered -- which the permutation check below
        //     establishes.
        if (indices != indicesBefore)
        {
            ReportFailure(name, "input indices modified by OptimizeFacesLRU");
            return;
        }

        // (2) Every entry is UNUSED32 or a valid face index; every valid face
        //     appears exactly once; invalid faces never appear.
        std::vector<uint32_t> seen(nFaces, 0);
        std::vector<uint32_t> optimizedOrder;
        for (size_t k = 0; k < nFaces; ++k)
        {
            const uint32_t r = remap[k];
            if (r == kUnusedFace)
                continue;

            if (r >= nFaces)
            {
                ReportFailure(name, "remap entry out of range");
                return;
            }
            if (!IsValidFace(indices.data(), r))
            {
                ReportFailure(name, "remap emitted a face with an unused index");
                return;
            }
            ++seen[r];
            optimizedOrder.push_back(r);
        }

        std::vector<uint32_t> sequentialOrder;
        bool permutationOk = true;
        for (uint32_t f = 0; f < nFaces; ++f)
        {
            if (!IsValidFace(indices.data(), f))
                continue;
            sequentialOrder.push_back(f);
            if (seen[f] != 1)
                permutationOk = false;
        }
        if (!permutationOk)
        {
            ReportFailure(name, "valid faces are not a permutation of the input");
            return;
        }

        // (4) Cache hit rate of the optimized order must be no worse than
        //     sequential traversal.
        const uint64_t optimizedMisses  = SimulateFifoMisses(indices.data(), optimizedOrder, lruCacheSize);
        const uint64_t sequentialMisses = SimulateFifoMisses(indices.data(), sequentialOrder, lruCacheSize);
        if (optimizedMisses > sequentialMisses)
        {
            std::printf("FAIL %-24s cache misses: optimized=%llu > sequential=%llu\n",
                name,
                static_cast<unsigned long long>(optimizedMisses),
                static_cast<unsigned long long>(sequentialMisses));
            ++g_failures;
            return;
        }

        std::printf("ok   %-24s faces=%2zu valid=%2zu misses opt=%llu seq=%llu\n",
            name,
            nFaces,
            sequentialOrder.size(),
            static_cast<unsigned long long>(optimizedMisses),
            static_cast<unsigned long long>(sequentialMisses));
    }

    // Deterministic pseudo-random degenerate meshes (fixed seed => fixed test).
    template<typename IndexT>
    void RunRandomCases(uint32_t seed, int caseCount, uint32_t maxVerts, uint32_t lruCacheSize)
    {
        uint32_t state = seed;
        auto next = [&]() -> uint32_t
        {
            state = state * 1664525u + 1013904223u;
            return state >> 8;
        };

        for (int c = 0; c < caseCount; ++c)
        {
            const uint32_t nFaces = 1 + next() % 48;
            const uint32_t nVerts = 1 + next() % maxVerts;

            std::vector<IndexT> indices;
            indices.reserve(size_t(nFaces) * 3);
            for (uint32_t f = 0; f < nFaces; ++f)
            {
                IndexT tri[3];
                for (uint32_t j = 0; j < 3; ++j)
                {
                    const uint32_t r = next() % 100;
                    if (r < 8)
                        tri[j] = kUnused<IndexT>; // unused index
                    else
                        tri[j] = IndexT(next() % nVerts);
                }
                // Frequently collapse corners to create zero-area triangles.
                if ((next() % 100) < 35)
                    tri[1] = tri[0];
                if ((next() % 100) < 20)
                    tri[2] = tri[0];
                // Occasionally duplicate the previous face exactly.
                if (f > 0 && (next() % 100) < 15)
                {
                    indices.push_back(indices[indices.size() - 3]);
                    indices.push_back(indices[indices.size() - 3]);
                    indices.push_back(indices[indices.size() - 3]);
                    continue;
                }
                indices.push_back(tri[0]);
                indices.push_back(tri[1]);
                indices.push_back(tri[2]);
            }

            char name[64];
            std::snprintf(name, sizeof(name), "random-%u-%d", seed, c);
            RunCase(name, indices, nVerts, lruCacheSize);
        }
    }

    template<typename IndexT>
    void FixedDegenerateMeshes()
    {
        const IndexT X = kUnused<IndexT>;

        RunCase<IndexT>("single-point", { IndexT(7), IndexT(7), IndexT(7) }, 8, 32);

        RunCase<IndexT>("all-degenerate",
            { 0, 0, 0, 1, 1, 2, 3, 4, 4, 5, 6, 5 }, 8, 32);

        RunCase<IndexT>("mixed",
            { 0, 1, 2, 2, 2, 3, 3, 4, 5, 4, 4, 4, 5, 0, 1, 1, 2, 1 }, 6, 32);

        RunCase<IndexT>("isolated-verts",
            { 0, 0, 0, 1, 2, 3, 3, 3, 1 }, 16, 32);

        RunCase<IndexT>("unused-faces",
            { 0, 1, 2, X, X, X, 2, 2, 1, X, X, X, 3, 4, 5 }, 6, 32);

        RunCase<IndexT>("partial-unused",
            { 0, X, 0, 0, 1, 2, 1, 1, X }, 3, 32);

        RunCase<IndexT>("duplicate-faces",
            { 0, 1, 2, 0, 1, 2, 2, 2, 0, 0, 1, 2, 3, 3, 3 }, 4, 32);

        RunCase<IndexT>("all-unused",
            { X, X, X, X, X, X }, 4, 32);

        RunCase<IndexT>("unused-around-valid",
            { X, X, X, 5, 5, 5, X, 1, X }, 8, 32);

        // Smallest and largest legal cache sizes.
        RunCase<IndexT>("cache-size-1",
            { 0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4 }, 8, 1);
        RunCase<IndexT>("cache-size-64",
            { 0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4 }, 8, 64);

        // A well-formed grid: every face valid, remap must be a full
        // permutation and the optimized order no worse than the input.
        {
            const uint32_t dim = 8;
            std::vector<IndexT> grid;
            for (uint32_t y = 0; y + 1 < dim; ++y)
            {
                for (uint32_t x = 0; x + 1 < dim; ++x)
                {
                    const IndexT v00 = IndexT(y * dim + x);
                    const IndexT v10 = IndexT(y * dim + x + 1);
                    const IndexT v01 = IndexT((y + 1) * dim + x);
                    const IndexT v11 = IndexT((y + 1) * dim + x + 1);
                    grid.insert(grid.end(), { v00, v10, v11, v00, v11, v01 });
                }
            }
            RunCase<IndexT>("grid-8x8", grid, dim * dim, 32);
        }
    }

    void FixedDegenerateMeshesEx()
    {
        // OptimizeFacesLRUEx: two attribute subsets, each with degenerate and
        // partially-unused faces.
        const std::vector<uint16_t> indices =
        {
            0, 1, 2, 2, 2, 0,             // subset 0
            3, 0xFFFF, 3, 4, 4, 4,        // subset 0
            5, 6, 7, 6, 6, 5,             // subset 1
            0xFFFF, 0xFFFF, 0xFFFF, 7, 5, 6, // subset 1
        };
        const std::vector<uint32_t> attributes = { 0, 0, 0, 0, 1, 1, 1, 1 };
        const size_t nFaces = indices.size() / 3;

        std::vector<uint32_t> remap(nFaces, 0xDEADBEEFu);
        const HRESULT hr = OptimizeFacesLRUEx(indices.data(), nFaces, 8, attributes.data(), remap.data(), 32);
        if (FAILED(hr))
        {
            ReportFailure("lru-ex-subsets", "OptimizeFacesLRUEx failed");
            return;
        }

        std::vector<uint32_t> seen(nFaces, 0);
        for (size_t k = 0; k < nFaces; ++k)
        {
            const uint32_t r = remap[k];
            if (r == kUnusedFace)
                continue;
            if (r >= nFaces || !IsValidFace(indices.data(), r))
            {
                ReportFailure("lru-ex-subsets", "invalid remap entry");
                return;
            }
            ++seen[r];
        }
        for (uint32_t f = 0; f < nFaces; ++f)
        {
            if (IsValidFace(indices.data(), f) && seen[f] != 1)
            {
                ReportFailure("lru-ex-subsets", "valid faces are not a permutation of the input");
                return;
            }
        }
        std::printf("ok   %-24s faces=%2zu\n", "lru-ex-subsets", nFaces);
    }
} // namespace

int main()
{
    FixedDegenerateMeshes<uint16_t>();
    FixedDegenerateMeshes<uint32_t>();

    // 32-bit indices with vertices beyond the 16-bit range.
    RunCase<uint32_t>("wide-indices",
        { 70000, 70000, 1, 1, 2, 70000, 0xFFFFFFFFu, 3, 3, 4, 5, 6 }, 70001, 32);

    FixedDegenerateMeshesEx();

    RunRandomCases<uint16_t>(1234, 200, 64, 32);
    RunRandomCases<uint16_t>(98765, 200, 300, 16);
    RunRandomCases<uint32_t>(555, 100, 1000, 32);

    if (g_failures)
    {
        std::printf("FAILED: %d case(s)\n", g_failures);
        return 1;
    }

    std::printf("PASSED\n");
    return 0;
}
