//-------------------------------------------------------------------------------------
// VertexCacheMissRateTest.cpp
//
// Regression test for ComputeVertexCacheMissRate.
//
// The post-transform vertex cache is driven by whole triangles. This test
// verifies that the library matches an independent FIFO reference simulator
// for:
//   * arbitrary index orderings, including triangle-strip / triangle-fan style
//     stitching (heavy vertex reuse, duplicated and zero-area faces);
//   * degenerate triangles that reference the "unused" sentinel index, either
//     in all three corners or only in some of them, interleaved with valid
//     faces;
//   * both 16-bit and 32-bit index buffers, including 32-bit-only vertices;
//   * the full range of legal cache sizes.
//
// It also pins the documented cube-mesh result from the wiki.
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

    template<typename IndexT>
    constexpr IndexT kUnused = IndexT(-1);

    void ReportFailure(const char* what)
    {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    }

    // Independent reference simulation of the post-transform FIFO vertex
    // cache. Faces containing the unused sentinel in any corner are not
    // submitted to the GPU and therefore never touch the cache; every other
    // face feeds its three corners into the FIFO in order. A miss evicts the
    // oldest entry (circular buffer), a hit leaves the cache untouched.
    template<typename IndexT>
    struct ReferenceResult
    {
        float acmr = -1.f;
        float atvr = -1.f;
        uint64_t misses = 0;
        size_t validFaces = 0;
    };

    template<typename IndexT>
    ReferenceResult<IndexT> ReferenceMissRate(const std::vector<IndexT>& indices,
        size_t nVerts,
        size_t cacheSize)
    {
        ReferenceResult<IndexT> result;

        const size_t nFaces = indices.size() / 3;
        if (!nFaces || !nVerts || !cacheSize)
            return result;

        std::vector<uint32_t> fifo(cacheSize, 0xFFFFFFFFu);
        size_t tail = 0;

        for (size_t face = 0; face < nFaces; ++face)
        {
            const IndexT i0 = indices[face * 3];
            const IndexT i1 = indices[face * 3 + 1];
            const IndexT i2 = indices[face * 3 + 2];

            if (i0 == kUnused<IndexT> || i1 == kUnused<IndexT> || i2 == kUnused<IndexT>)
                continue;

            ++result.validFaces;

            const IndexT tri[3] = { i0, i1, i2 };
            for (size_t j = 0; j < 3; ++j)
            {
                bool found = false;
                for (size_t ptr = 0; ptr < cacheSize; ++ptr)
                {
                    if (fifo[ptr] == static_cast<uint32_t>(tri[j]))
                    {
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    ++result.misses;
                    fifo[tail] = static_cast<uint32_t>(tri[j]);
                    ++tail;
                    if (tail == cacheSize)
                        tail = 0;
                }
            }
        }

        result.acmr = result.validFaces ? float(result.misses) / float(result.validFaces) : 0.f;
        result.atvr = float(result.misses) / float(nVerts);
        return result;
    }

    template<typename IndexT>
    bool RunComparison(const char* name,
        const std::vector<IndexT>& indices,
        size_t nVerts,
        size_t cacheSize)
    {
        const size_t nFaces = indices.size() / 3;

        float acmr = -1.f;
        float atvr = -1.f;
        ComputeVertexCacheMissRate(indices.data(), nFaces, nVerts, cacheSize, acmr, atvr);

        const ReferenceResult<IndexT> ref = ReferenceMissRate(indices, nVerts, cacheSize);

        if (acmr != ref.acmr || atvr != ref.atvr)
        {
            std::printf("FAIL %-22s cache=%2zu got acmr=%.8g atvr=%.8g, want acmr=%.8g atvr=%.8g\n",
                name,
                cacheSize,
                acmr,
                atvr,
                ref.acmr,
                ref.atvr);
            ++g_failures;
            return false;
        }
        return true;
    }

    // Deterministic pseudo-random meshes (fixed seeds => fixed test data).
    // The generator deliberately produces strip/fan-like orderings (shared
    // corners), zero-area faces, duplicated faces, fully and partially unused
    // faces, and arbitrary per-face corner orderings.
    template<typename IndexT>
    void RunRandomCases(uint32_t seed, int caseCount, uint32_t maxVerts, size_t cacheSize)
    {
        uint32_t state = seed;
        auto next = [&]() -> uint32_t
        {
            state = state * 1664525u + 1013904223u;
            return state >> 8;
        };

        for (int c = 0; c < caseCount; ++c)
        {
            const uint32_t nFaces = 1 + next() % 64;
            const uint32_t nVerts = 1 + next() % maxVerts;

            std::vector<IndexT> indices;
            indices.reserve(size_t(nFaces) * 3);

            for (uint32_t f = 0; f < nFaces; ++f)
            {
                IndexT tri[3];
                for (uint32_t j = 0; j < 3; ++j)
                {
                    const uint32_t r = next() % 100;
                    if (r < 10)
                        tri[j] = kUnused<IndexT>; // unused corner / face
                    else
                        tri[j] = IndexT(next() % nVerts);
                }

                // Strip/fan stitching: reuse a corner of the previous face.
                if (f > 0 && (next() % 100) < 40)
                {
                    tri[next() % 3] = indices[indices.size() - 1 - (next() % 3)];
                }

                // Collapse corners to create zero-area (but non-sentinel) faces.
                if ((next() % 100) < 35)
                    tri[1] = tri[0];
                if ((next() % 100) < 20)
                    tri[2] = tri[0];

                // Occasionally duplicate the previous face exactly.
                if (f > 0 && (next() % 100) < 15)
                {
                    indices.push_back(indices[indices.size() - 3]);
                    indices.push_back(indices[indices.size() - 2]);
                    indices.push_back(indices[indices.size() - 1]);
                    continue;
                }

                indices.push_back(tri[0]);
                indices.push_back(tri[1]);
                indices.push_back(tri[2]);
            }

            char name[64];
            std::snprintf(name, sizeof(name), "random-%u-%d", seed, c);
            RunComparison(name, indices, nVerts, cacheSize);
        }
    }

    template<typename IndexT>
    void FixedCases()
    {
        const IndexT X = kUnused<IndexT>;

        // Documented wiki cube: acmr == 2.f, atvr == 1.f at cache size 12.
        {
            const std::vector<IndexT> cube =
            {
                3, 1, 0, 2, 1, 3,
                6, 4, 5, 7, 4, 6,
                11, 9, 8, 10, 9, 11,
                14, 12, 13, 15, 12, 14,
                19, 17, 16, 18, 17, 19,
                22, 20, 21, 23, 20, 22
            };
            float acmr = -1.f;
            float atvr = -1.f;
            ComputeVertexCacheMissRate(cube.data(), cube.size() / 3, 24, 12, acmr, atvr);
            if (acmr != 2.f || atvr != 1.f)
            {
                std::printf("FAIL cube-golden            got acmr=%g atvr=%g, want 2 / 1\n", acmr, atvr);
                ++g_failures;
            }
        }

        // Fully-unused faces must not touch the cache at all.
        RunComparison("all-unused",
            std::vector<IndexT>{ X, X, X, X, X, X }, 4, 12);

        // Partially-unused face interleaved with valid faces: the sentinel
        // face must not perturb the FIFO state seen by the surrounding faces.
        RunComparison("partial-unused",
            std::vector<IndexT>{ 0, X, 0, 0, 1, 2, 1, 1, X, 2, 1, 0 }, 3, 12);

        RunComparison("unused-around-valid",
            std::vector<IndexT>{ X, X, X, 5, 5, 5, X, 1, X, 0, 1, 2 }, 8, 12);

        // Strip stitching: runs of zero-area connector faces between fans.
        RunComparison("strip-stitch",
            std::vector<IndexT>{ 0, 1, 2, 2, 2, 3, 3, 4, 5, 5, 5, 5, 6, 7, 8 }, 9, 12);

        // All non-sentinel degenerate faces (still rendered, still simulated).
        RunComparison("all-degenerate",
            std::vector<IndexT>{ 0, 0, 0, 1, 1, 2, 3, 4, 4, 5, 6, 5 }, 8, 12);

        RunComparison("duplicate-faces",
            std::vector<IndexT>{ 0, 1, 2, 0, 1, 2, 2, 2, 0, 0, 1, 2, 3, 3, 3 }, 4, 12);

        // Single triangle: always 3 misses => acmr 3, atvr depends on nVerts.
        RunComparison("single-tri",
            std::vector<IndexT>{ 0, 1, 2 }, 3, 12);

        // Cache of one: only back-to-back identical corners can hit.
        RunComparison("cache-size-1",
            std::vector<IndexT>{ 0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4 }, 8, 1);

        // Extreme legal cache sizes.
        RunComparison("cache-size-64",
            std::vector<IndexT>{ 0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4 }, 8, 64);

        // Empty / invalid argument combinations stay at the -1 sentinel.
        {
            float acmr = 42.f;
            float atvr = 42.f;
            const std::vector<IndexT> one = { 0, 1, 2 };
            ComputeVertexCacheMissRate(one.data(), 1, 3, 0, acmr, atvr);
            if (acmr != -1.f || atvr != -1.f)
            {
                ReportFailure("zero cache size must return -1");
            }

            acmr = 42.f;
            atvr = 42.f;
            ComputeVertexCacheMissRate(static_cast<const IndexT*>(nullptr), 1, 3, 12, acmr, atvr);
            if (acmr != -1.f || atvr != -1.f)
            {
                ReportFailure("null indices must return -1");
            }
        }

        // Out-of-range index is an input error: outputs stay at -1.
        {
            float acmr = 42.f;
            float atvr = 42.f;
            const std::vector<IndexT> bad = { 0, 1, IndexT(9) };
            ComputeVertexCacheMissRate(bad.data(), 1, 3, 12, acmr, atvr);
            if (acmr != -1.f || atvr != -1.f)
            {
                ReportFailure("out-of-range index must return -1");
            }
        }
    }

    template<typename IndexT>
    void WideIndexCase()
    {
        // Vertices that cannot be addressed with 16-bit indices.
        const std::vector<IndexT> wide =
        {
            70000, 70000, 1,
            1, 2, 70000,
            IndexT(0xFFFFFFFFu), 3, 3,
            4, 5, 6
        };
        for (size_t cache : { size_t(1), size_t(12), size_t(32), size_t(64) })
        {
            char name[64];
            std::snprintf(name, sizeof(name), "wide-indices-c%zu", cache);
            RunComparison(name, wide, 70001, cache);
        }
    }
} // namespace

int main()
{
    FixedCases<uint16_t>();
    FixedCases<uint32_t>();

    WideIndexCase<uint32_t>();

    // Random orderings across index widths, mesh sizes and cache sizes.
    RunRandomCases<uint16_t>(1234, 200, 64, 12);
    RunRandomCases<uint16_t>(98765, 200, 300, 16);
    RunRandomCases<uint16_t>(4242, 150, 8, 1);
    RunRandomCases<uint32_t>(555, 200, 1000, 32);
    RunRandomCases<uint32_t>(31337, 150, 100000, 64);

    if (g_failures)
    {
        std::printf("FAILED: %d case(s)\n", g_failures);
        return 1;
    }

    std::printf("PASSED\n");
    return 0;
}
