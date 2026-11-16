//-------------------------------------------------------------------------------------
// DirectXMeshOptimizeLRU.cpp
//
// DirectX Mesh Geometry Library - Mesh optimization
//
// Forsyth "Linear-Speed Vertex Cache Optimisation"
// https://tomforsyth1000.github.io/papers/fast_vert_cache_opt.html
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// https://go.microsoft.com/fwlink/?LinkID=324981
//-------------------------------------------------------------------------------------

#include "DirectXMeshP.h"

using namespace DirectX;

namespace
{
    // This code was authored and released into the public domain by Adrian Stone (stone@gameangst.com).

    // code for computing vertex score was taken, as much as possible
    // directly from the original publication.
    float ComputeVertexCacheScore(uint32_t cachePosition, uint32_t vertexCacheSize) noexcept
    {
        constexpr float FindVertexScore_CacheDecayPower = 1.5f;
        constexpr float FindVertexScore_LastTriScore    = 0.75f;

        float score = 0.0f;
        if (cachePosition >= vertexCacheSize)
        {
            // Vertex is not in FIFO cache - no score.
        }
        else
        {
            if (cachePosition < 3)
            {
                // This vertex was used in the last triangle,
                // so it has a fixed score, whichever of the three
                // it's in. Otherwise, you can get very different
                // answers depending on whether you add
                // the triangle 1,2,3 or 3,1,2 - which is silly.

                score = FindVertexScore_LastTriScore;
            }
            else
            {
                // Points for being high in the cache.
                const float scaler = 1.0f / float(vertexCacheSize - 3u);
                score              = 1.0f - float(cachePosition - 3u) * scaler;
                score              = powf(score, FindVertexScore_CacheDecayPower);
            }
        }

        return score;
    }

    float ComputeVertexValenceScore(uint32_t numActiveFaces) noexcept
    {
        constexpr float FindVertexScore_ValenceBoostScale = 2.0f;
        constexpr float FindVertexScore_ValenceBoostPower = 0.5f;

        float score = 0.f;

        // Bonus points for having a low number of tris still to
        // use the vert, so we get rid of lone verts quickly.
        const float valenceBoost = powf(static_cast<float>(numActiveFaces), -FindVertexScore_ValenceBoostPower);

        score += FindVertexScore_ValenceBoostScale * valenceBoost;
        return score;
    }

    enum
    {
        kMaxVertexCacheSize = 64
    };
    enum
    {
        kMaxPrecomputedVertexValenceScores = 64
    };

    float s_vertexCacheScores[kMaxVertexCacheSize + 1][kMaxVertexCacheSize];
    float s_vertexValenceScores[kMaxPrecomputedVertexValenceScores];

#ifdef _WIN32
    static INIT_ONCE s_initOnce = INIT_ONCE_STATIC_INIT;

    BOOL WINAPI ComputeVertexScores(PINIT_ONCE, PVOID, PVOID*) noexcept
#else
    std::once_flag s_initOnce;

    void ComputeVertexScores() noexcept
#endif
    {
        for (uint32_t cacheSize = 0; cacheSize <= kMaxVertexCacheSize; ++cacheSize)
        {
            for (uint32_t cachePos = 0; cachePos < cacheSize; ++cachePos)
            {
                s_vertexCacheScores[cacheSize][cachePos] = ComputeVertexCacheScore(cachePos, cacheSize);
            }
        }

        for (uint32_t valence = 0; valence < kMaxPrecomputedVertexValenceScores; ++valence)
        {
            s_vertexValenceScores[valence] = ComputeVertexValenceScore(valence);
        }

#ifdef _WIN32
        return TRUE;
#endif
    }

    float FindVertexScore(uint32_t numActiveFaces, uint32_t cachePosition, uint32_t vertexCacheSize) noexcept
    {
        if (numActiveFaces == 0)
        {
            // No tri needs this vertex!

            return -1.0f;
        }

        float score = 0.f;

        if (cachePosition < vertexCacheSize)
        {
            score += s_vertexCacheScores[vertexCacheSize][cachePosition];
        }

        if (numActiveFaces < kMaxPrecomputedVertexValenceScores)
        {
            score += s_vertexValenceScores[numActiveFaces];
        }
        else
        {
            score += ComputeVertexValenceScore(numActiveFaces);
        }

        return score;
    }

    template<typename IndexType>
    struct OptimizeVertexData
    {
        float     score;
        uint32_t  activeFaceListStart;
        uint32_t  activeFaceListSize;
        IndexType cachePos0;
        IndexType cachePos1;

        OptimizeVertexData() noexcept
            : score(0.f),
              activeFaceListStart(0),
              activeFaceListSize(0),
              cachePos0(0),
              cachePos1(0)
        {}
    };

    template<typename T, typename IndexType>

    struct IndexSortCompareIndexed
    {
        const IndexType* _indexData;

        IndexSortCompareIndexed(const IndexType* indexData) noexcept
            : _indexData(indexData)
        {}

        bool operator()(T a, T b) const noexcept
        {
            IndexType indexA = _indexData[a];
            IndexType indexB = _indexData[b];

            if (indexA < indexB)
                return true;

            return false;
        }
    };

    template<typename T, typename IndexType>
    struct FaceValenceSort
    {
        const OptimizeVertexData<IndexType>* _vertexData;

        FaceValenceSort(const OptimizeVertexData<IndexType>* vertexData) noexcept
            : _vertexData(vertexData)
        {}

        bool operator()(T a, T b) const noexcept
        {
            const OptimizeVertexData<IndexType>* vA = _vertexData + size_t(a) * 3;
            const OptimizeVertexData<IndexType>* vB = _vertexData + size_t(b) * 3;

            const uint32_t aValence = vA[0].activeFaceListSize + vA[1].activeFaceListSize + vA[2].activeFaceListSize;
            const uint32_t bValence = vB[0].activeFaceListSize + vB[1].activeFaceListSize + vB[2].activeFaceListSize;

            // higher scoring faces are those with lower valence totals

            // reverse sort (reverse of reverse)

            if (aValence < bValence)
                return true;

            return false;
        }
    };

    template<typename IndexType>
    HRESULT OptimizeFacesImpl(_In_reads_(indexCount) const IndexType* indexList,
        uint32_t                                                      indexCount,
        size_t                                                        nVerts,
        _Out_writes_(indexCount / 3) uint32_t*                        faceRemap,
        uint32_t                                                      lruCacheSize,
        uint32_t                                                      offset)
    {
        std::unique_ptr<OptimizeVertexData<IndexType>[]> vertexDataList(new (std::nothrow) OptimizeVertexData<IndexType>[indexCount]);
        if (!vertexDataList)
            return E_OUTOFMEMORY;

        std::unique_ptr<uint32_t[]> vertexRemap(new (std::nothrow) uint32_t[indexCount]);
        std::unique_ptr<uint32_t[]> activeFaceList(new (std::nothrow) uint32_t[indexCount]);
        if (!vertexRemap || !activeFaceList)
            return E_OUTOFMEMORY;

        const uint32_t faceCount = indexCount / 3;

        std::unique_ptr<uint8_t[]>  processedFaceList(new (std::nothrow) uint8_t[faceCount]);
        std::unique_ptr<uint32_t[]> faceSorted(new (std::nothrow) uint32_t[faceCount]);
        std::unique_ptr<uint32_t[]> faceReverseLookup(new (std::nothrow) uint32_t[faceCount]);
        if (!processedFaceList || !faceSorted || !faceReverseLookup)
            return E_OUTOFMEMORY;

        memset(processedFaceList.get(), 0, sizeof(uint8_t) * faceCount);

        // build the vertex remap table
        uint32_t uniqueVertexCount = 0;
        {
            using indexSorter = IndexSortCompareIndexed<uint32_t, IndexType>;

            std::unique_ptr<uint32_t[]> indexSorted(new (std::nothrow) uint32_t[indexCount]);
            if (!indexSorted)
                return E_OUTOFMEMORY;

            for (uint32_t i = 0; i < indexCount; i++)
            {
                indexSorted[i] = i;
            }

            const indexSorter sortFunc(indexList);
            std::sort(indexSorted.get(), indexSorted.get() + indexCount, sortFunc);

            bool first = false;
            for (uint32_t i = 0; i < indexCount; i++)
            {
                const uint32_t idx = indexSorted[i];
                if (indexList[idx] == IndexType(-1))
                {
                    vertexRemap[idx] = UNUSED32;
                    continue;
                }

                if (indexList[idx] >= nVerts)
                    return E_UNEXPECTED;

                if (!i || first || sortFunc(indexSorted[i - 1], idx))
                {
                    // it's not a duplicate
                    vertexRemap[idx] = uniqueVertexCount;
                    uniqueVertexCount++;
                    first = false;
                }
                else
                {
                    vertexRemap[indexSorted[i]] = vertexRemap[indexSorted[i - 1]];
                }
            }
        }

        // compute face count per vertex
        //
        // Faces that reference one or more unused vertices (i.e. the index
        // value IndexType(-1)) can never be emitted by the optimizer, so they
        // must not contribute to any vertex's active face list. Otherwise the
        // stale entries would later be dereferenced through vertexRemap as
        // vertexDataList[UNUSED32], which is far out of bounds.
        uint32_t ignoredCorners = 0;
        for (uint32_t i = 0; i < indexCount; i += 3)
        {
            if (vertexRemap[i] == UNUSED32
                || vertexRemap[size_t(i) + 1] == UNUSED32
                || vertexRemap[size_t(i) + 2] == UNUSED32)
            {
                ignoredCorners += 3;
                continue;
            }

            for (uint32_t j = 0; j < 3; ++j)
            {
                OptimizeVertexData<IndexType>& vertexData = vertexDataList[vertexRemap[size_t(i) + size_t(j)]];
                vertexData.activeFaceListSize++;
            }
        }

        constexpr IndexType kEvictedCacheIndex = std::numeric_limits<IndexType>::max();
        {
            // allocate face list per vertex
            uint32_t curActiveFaceListPos = 0;
            for (uint32_t i = 0; i < uniqueVertexCount; ++i)
            {
                OptimizeVertexData<IndexType>& vertexData = vertexDataList[i];
                vertexData.cachePos0                      = kEvictedCacheIndex;
                vertexData.cachePos1                      = kEvictedCacheIndex;
                vertexData.activeFaceListStart            = curActiveFaceListPos;
                curActiveFaceListPos += vertexData.activeFaceListSize;
                vertexData.score = FindVertexScore(vertexData.activeFaceListSize, vertexData.cachePos0, lruCacheSize);

                vertexData.activeFaceListSize = 0;
            }

#ifndef NDEBUG
            assert(curActiveFaceListPos == (indexCount - ignoredCorners));
#else
            std::ignore = ignoredCorners;
#endif
        }

        // sort unprocessed faces by highest score
        for (uint32_t f = 0; f < faceCount; f++)
        {
            faceSorted[f] = f;
        }

        const FaceValenceSort<uint32_t, IndexType> faceValenceSort(vertexDataList.get());
        std::sort(faceSorted.get(), faceSorted.get() + faceCount, faceValenceSort);

        for (uint32_t f = 0; f < faceCount; f++)
        {
            faceReverseLookup[faceSorted[f]] = f;
        }

        // fill out face list per vertex
        for (uint32_t i = 0; i < indexCount; i += 3)
        {
            if (vertexRemap[i] == UNUSED32
                || vertexRemap[size_t(i) + 1] == UNUSED32
                || vertexRemap[size_t(i) + 2] == UNUSED32)
            {
                // skipped face (see above)
                continue;
            }

            for (uint32_t j = 0; j < 3; ++j)
            {
                const uint32_t v = vertexRemap[size_t(i) + size_t(j)];

                OptimizeVertexData<IndexType>& vertexData                                              = vertexDataList[v];
                activeFaceList[size_t(vertexData.activeFaceListStart) + vertexData.activeFaceListSize] = i;
                vertexData.activeFaceListSize++;
            }
        }

        uint32_t  vertexCacheBuffer[(kMaxVertexCacheSize + 3) * 2] = {};
        uint32_t* cache0                                           = vertexCacheBuffer;
        uint32_t* cache1                                           = vertexCacheBuffer + (kMaxVertexCacheSize + 3);
        uint32_t  entriesInCache0                                  = 0;

        uint32_t bestFace = 0;
        for (size_t i = 0; i < indexCount; i += 3)
        {
            if (vertexRemap[i] == UNUSED32 || vertexRemap[i + 1] == UNUSED32 || vertexRemap[i + 2] == UNUSED32)
            {
                continue;
            }

            bestFace = static_cast<uint32_t>(i);
            break;
        }

        float bestScore = -1.f;

        uint32_t nextBestFace = 0;

        uint32_t curFace = 0;
        for (size_t i = 0; i < indexCount; i += 3)
        {
            if (vertexRemap[i] == UNUSED32 || vertexRemap[i + 1] == UNUSED32 || vertexRemap[i + 2] == UNUSED32)
            {
                continue;
            }

            if (bestScore < 0.f)
            {
                // no verts in the cache are used by any unprocessed faces so
                // search all unprocessed faces for a new starting point
                while (nextBestFace < faceCount)
                {
                    const uint32_t faceIndex = faceSorted[nextBestFace++];
                    if (processedFaceList[faceIndex] == 0)
                    {
                        const uint32_t face = faceIndex * 3;
                        const uint32_t i0   = vertexRemap[face];
                        const uint32_t i1   = vertexRemap[size_t(face) + 1];
                        const uint32_t i2   = vertexRemap[size_t(face) + 2];
                        if (i0 != UNUSED32 && i1 != UNUSED32 && i2 != UNUSED32)
                        {
                            // we're searching a pre-sorted list, first one we find will be the best
                            bestFace  = face;
                            bestScore = vertexDataList[i0].score + vertexDataList[i1].score + vertexDataList[i2].score;
                            break;
                        }
                    }
                }
                assert(bestScore >= 0.f);
                if (bestScore < 0.f)
                {
                    // No unprocessed face with valid vertices remains. Bail out
                    // rather than re-emitting a stale bestFace (which would
                    // corrupt the output remap and walk off the active lists).
                    break;
                }
            }

            processedFaceList[bestFace / 3] = 1;
            uint16_t entriesInCache1        = 0;

            faceRemap[curFace] = (bestFace / 3) + offset;
            curFace++;

            // add bestFace to LRU cache
            assert(vertexRemap[bestFace] != UNUSED32);
            assert(vertexRemap[size_t(bestFace) + 1] != UNUSED32);
            assert(vertexRemap[size_t(bestFace) + 2] != UNUSED32);

            for (size_t v = 0; v < 3; ++v)
            {
                OptimizeVertexData<IndexType>& vertexData = vertexDataList[vertexRemap[bestFace + v]];

                if (vertexData.cachePos1 >= entriesInCache1)
                {
                    vertexData.cachePos1      = entriesInCache1;
                    cache1[entriesInCache1++] = vertexRemap[bestFace + v];

                    if (vertexData.activeFaceListSize == 1)
                    {
                        --vertexData.activeFaceListSize;
                        continue;
                    }
                }

                assert(vertexData.activeFaceListSize > 0);
                uint32_t* begin = activeFaceList.get() + vertexData.activeFaceListStart;
                uint32_t* end   = activeFaceList.get() + (size_t(vertexData.activeFaceListStart) + vertexData.activeFaceListSize);
                uint32_t* it    = std::find(begin, end, bestFace);

                assert(it != end);
                if (it == end)
                {
                    // Degenerate input left this face out of the vertex's active
                    // list; skip it instead of swapping with end[-1] on what may
                    // be an empty range (undefined behavior).
                    continue;
                }

                std::swap(*it, *(end - 1));

                --vertexData.activeFaceListSize;
                vertexData.score = FindVertexScore(vertexData.activeFaceListSize, vertexData.cachePos1, lruCacheSize);

                // need to re-sort the faces that use this vertex, as their score will change due to activeFaceListSize shrinking
                for (const uint32_t* fi = begin; fi != end - 1; ++fi)
                {
                    const uint32_t faceIndex = *fi / 3;
                    uint32_t       n         = faceReverseLookup[faceIndex];
                    assert(faceSorted[n] == faceIndex);

                    // found it, now move it up
                    while (n > 0)
                    {
                        if (faceValenceSort(n, n - 1))
                        {
                            faceReverseLookup[faceSorted[n]]     = n - 1;
                            faceReverseLookup[faceSorted[n - 1]] = n;
                            std::swap(faceSorted[n], faceSorted[n - 1]);
                            n--;
                        }
                        else
                        {
                            break;
                        }
                    }
                }
            }

            // move the rest of the old verts in the cache down and compute their new scores
            for (uint32_t c0 = 0; c0 < entriesInCache0; ++c0)
            {
                OptimizeVertexData<IndexType>& vertexData = vertexDataList[cache0[c0]];

                if (vertexData.cachePos1 >= entriesInCache1)
                {
                    vertexData.cachePos1      = entriesInCache1;
                    cache1[entriesInCache1++] = cache0[c0];
                    vertexData.score          = FindVertexScore(vertexData.activeFaceListSize, vertexData.cachePos1, lruCacheSize);

                    // don't need to re-sort this vertex... once it gets out of the cache, it'll have its original score
                }
            }

            // find the best scoring triangle in the current cache (including up to 3 that were just evicted)
            bestScore = -1.f;

            for (uint32_t c1 = 0; c1 < entriesInCache1; ++c1)
            {
                OptimizeVertexData<IndexType>& vertexData = vertexDataList[cache1[c1]];
                vertexData.cachePos0                      = vertexData.cachePos1;
                vertexData.cachePos1                      = kEvictedCacheIndex;

                for (uint32_t j = 0; j < vertexData.activeFaceListSize; ++j)
                {
                    const uint32_t face      = activeFaceList[size_t(vertexData.activeFaceListStart) + j];
                    float          faceScore = 0.f;

                    for (uint32_t v = 0; v < 3; v++)
                    {
                        const OptimizeVertexData<IndexType>& faceVertexData = vertexDataList[vertexRemap[size_t(face) + v]];
                        faceScore += faceVertexData.score;
                    }

                    if (faceScore > bestScore)
                    {
                        bestScore = faceScore;
                        bestFace  = face;
                    }
                }
            }

            std::swap(cache0, cache1);

            entriesInCache0 = std::min<uint32_t>(entriesInCache1, lruCacheSize);
        }

        // For adversarial or degenerate inputs the greedy heuristic can produce
        // an ordering whose vertex-cache hit rate is worse than simply walking
        // the faces in their original order. Simulate a FIFO cache of
        // lruCacheSize entries for both orderings and fall back to the original
        // order if the optimized one is worse.
        auto countCacheMisses = [&](const uint32_t* orderedFaces, uint32_t orderedCount) -> uint64_t
        {
            uint32_t fifo[kMaxVertexCacheSize] = {};
            uint32_t fifoSize                  = 0;
            uint32_t fifoHead                  = 0;
            uint64_t misses                    = 0;

            for (uint32_t k = 0; k < orderedCount; ++k)
            {
                const size_t face = size_t(orderedFaces[k]) * 3;
                for (size_t j = 0; j < 3; ++j)
                {
                    const uint32_t v = vertexRemap[face + j];

                    bool hit = false;
                    for (uint32_t c = 0; c < fifoSize; ++c)
                    {
                        if (fifo[c] == v)
                        {
                            hit = true;
                            break;
                        }
                    }
                    if (hit)
                        continue;

                    ++misses;
                    if (fifoSize < lruCacheSize)
                    {
                        fifo[fifoSize++] = v;
                    }
                    else
                    {
                        fifo[fifoHead] = v;
                        fifoHead       = (fifoHead + 1) % lruCacheSize;
                    }
                }
            }

            return misses;
        };

        // faceSorted/faceReverseLookup are no longer needed by the optimizer;
        // reuse them to hold the sequential and optimized face orders.
        uint32_t validFaceCount = 0;
        for (uint32_t f = 0; f < faceCount; ++f)
        {
            const size_t face = size_t(f) * 3;
            if (vertexRemap[face] == UNUSED32
                || vertexRemap[face + 1] == UNUSED32
                || vertexRemap[face + 2] == UNUSED32)
            {
                continue;
            }

            faceSorted[validFaceCount++] = f;
        }

        assert(validFaceCount == curFace);

        bool useOriginalOrder = (curFace != validFaceCount);
        if (!useOriginalOrder)
        {
            for (uint32_t k = 0; k < curFace; ++k)
            {
                faceReverseLookup[k] = faceRemap[k] - offset;
            }

            useOriginalOrder = countCacheMisses(faceReverseLookup.get(), curFace) > countCacheMisses(faceSorted.get(), validFaceCount);
        }

        if (useOriginalOrder)
        {
            // The original order is at least as good; emit it instead.
            curFace = 0;
            for (uint32_t k = 0; k < validFaceCount; ++k)
            {
                faceRemap[curFace++] = faceSorted[k] + offset;
            }
        }

        for (; curFace < faceCount; ++curFace)
        {
            faceRemap[curFace] = UNUSED32;
        }

        return S_OK;
    }
} // namespace

//=====================================================================================
// Entry-points
//=====================================================================================

_Use_decl_annotations_ HRESULT
DirectX::OptimizeFacesLRU(const uint16_t* indices, size_t nFaces, size_t nVerts, uint32_t* faceRemap, uint32_t lruCacheSize)
{
    if (!indices || !nFaces || !nVerts || !faceRemap)
        return E_INVALIDARG;

    if (!lruCacheSize || lruCacheSize > kMaxVertexCacheSize)
        return E_INVALIDARG;

    if ((uint64_t(nFaces) * 3) >= UINT32_MAX)
        return HRESULT_E_ARITHMETIC_OVERFLOW;

#ifdef _WIN32
    InitOnceExecuteOnce(&s_initOnce, ComputeVertexScores, nullptr, nullptr);
#else
    std::call_once(s_initOnce, ComputeVertexScores);
#endif

    return OptimizeFacesImpl<uint16_t>(indices, static_cast<uint32_t>(nFaces * 3), nVerts, faceRemap, lruCacheSize, 0);
}

_Use_decl_annotations_ HRESULT
DirectX::OptimizeFacesLRU(const uint32_t* indices, size_t nFaces, size_t nVerts, uint32_t* faceRemap, uint32_t lruCacheSize)
{
    if (!indices || !nFaces || !nVerts || !faceRemap)
        return E_INVALIDARG;

    if (!lruCacheSize || lruCacheSize > kMaxVertexCacheSize)
        return E_INVALIDARG;

    if ((uint64_t(nFaces) * 3) >= UINT32_MAX)
        return HRESULT_E_ARITHMETIC_OVERFLOW;

#ifdef _WIN32
    InitOnceExecuteOnce(&s_initOnce, ComputeVertexScores, nullptr, nullptr);
#else
    std::call_once(s_initOnce, ComputeVertexScores);
#endif

    return OptimizeFacesImpl<uint32_t>(indices, static_cast<uint32_t>(nFaces * 3), nVerts, faceRemap, lruCacheSize, 0);
}

//-------------------------------------------------------------------------------------
_Use_decl_annotations_ HRESULT DirectX::OptimizeFacesLRUEx(const uint16_t* indices,
    size_t                                                                 nFaces,
    size_t                                                                 nVerts,
    const uint32_t*                                                        attributes,
    uint32_t*                                                              faceRemap,
    uint32_t                                                               lruCacheSize)
{
    if (!indices || !nFaces || !nVerts || !attributes || !faceRemap)
        return E_INVALIDARG;

    if (!lruCacheSize || lruCacheSize > kMaxVertexCacheSize)
        return E_INVALIDARG;

    if ((uint64_t(nFaces) * 3) >= UINT32_MAX)
        return HRESULT_E_ARITHMETIC_OVERFLOW;

#ifdef _WIN32
    InitOnceExecuteOnce(&s_initOnce, ComputeVertexScores, nullptr, nullptr);
#else
    std::call_once(s_initOnce, ComputeVertexScores);
#endif

    auto subsets = ComputeSubsets(attributes, nFaces);

    if (subsets.empty())
        return E_UNEXPECTED;

    memset(faceRemap, 0, sizeof(uint32_t) * nFaces);

    for (const auto& it : subsets)
    {
        if (it.first >= nFaces)
            return E_UNEXPECTED;

        if ((uint64_t(it.first) + uint64_t(it.second)) >= UINT32_MAX)
            return HRESULT_E_ARITHMETIC_OVERFLOW;

        const uint32_t faceMax = uint32_t(it.first + it.second);

        if (faceMax > nFaces)
            return E_UNEXPECTED;

        HRESULT hr = OptimizeFacesImpl<uint16_t>(&indices[it.first * 3],
            static_cast<uint32_t>(it.second * 3),
            nVerts,
            &faceRemap[it.first],
            lruCacheSize,
            uint32_t(it.first));
        if (FAILED(hr))
            return hr;
    }

    return S_OK;
}

_Use_decl_annotations_ HRESULT DirectX::OptimizeFacesLRUEx(const uint32_t* indices,
    size_t                                                                 nFaces,
    size_t                                                                 nVerts,
    const uint32_t*                                                        attributes,
    uint32_t*                                                              faceRemap,
    uint32_t                                                               lruCacheSize)
{
    if (!indices || !nFaces || !nVerts || !attributes || !faceRemap)
        return E_INVALIDARG;

    if (!lruCacheSize || lruCacheSize > kMaxVertexCacheSize)
        return E_INVALIDARG;

    if ((uint64_t(nFaces) * 3) >= UINT32_MAX)
        return HRESULT_E_ARITHMETIC_OVERFLOW;

#ifdef _WIN32
    InitOnceExecuteOnce(&s_initOnce, ComputeVertexScores, nullptr, nullptr);
#else
    std::call_once(s_initOnce, ComputeVertexScores);
#endif

    auto subsets = ComputeSubsets(attributes, nFaces);

    if (subsets.empty())
        return E_UNEXPECTED;

    memset(faceRemap, 0, sizeof(uint32_t) * nFaces);

    for (const auto& it : subsets)
    {
        if (it.first >= nFaces)
            return E_UNEXPECTED;

        if ((uint64_t(it.first) + uint64_t(it.second)) >= UINT32_MAX)
            return HRESULT_E_ARITHMETIC_OVERFLOW;

        const uint32_t faceMax = uint32_t(it.first + it.second);

        if (faceMax > nFaces)
            return E_UNEXPECTED;

        HRESULT hr = OptimizeFacesImpl<uint32_t>(&indices[it.first * 3],
            static_cast<uint32_t>(it.second * 3),
            nVerts,
            &faceRemap[it.first],
            lruCacheSize,
            uint32_t(it.first));
        if (FAILED(hr))
            return hr;
    }

    return S_OK;
}
