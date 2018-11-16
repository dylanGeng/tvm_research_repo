```
#ifndef __ROIPOOLING_SCENARIO_128_C0__
#define __ROIPOOLING_SCENARIO_128_C0__

#include "roipooling_perf_fwd_executor.hpp"
#include "sync_common.hpp"
#include "executor_common.hpp"

namespace cce
{
namespace kernel
{
inline __aicore__ void roipoolingPerf128C0(__ubuf__ void *flowTableAddr, int64_t isLastBatch)
{
    __ubuf__ kRoipoolingPerfFlowTable_t *flowTable = (__ubuf__ kRoipoolingPerfFlowTable_t *)flowTableAddr;
    __ubuf__ kRoiMulitBatchParams_t *roiParams = &flowTable->roiMultiBatchParams[isLastBatch];

    int64_t pooledH = flowTable->pooledH;
    int64_t pooledW = flowTable->pooledW;
    int64_t inputH = flowTable->inputH;
    int64_t inputW = flowTable->inputW;

    uint64_t outCiOffset = flowTable->outCiOffset;
    uint64_t roiOutOffset = flowTable->roiOutOffset;
    uint64_t roiGroupOffset = flowTable->roiGroupOffset;

    //data Addrs
    __gm__ half *roiAddr = (__gm__ half *)flowTable->roiAddr;
    __gm__ half *inputAddr = (__gm__ half *)flowTable->inputAddr;
    __gm__ half *outputAddr = (__gm__ half *)flowTable->outputAddr;

    uint64_t roiNCHWConvXt = flowTable->roiNCHWConvXt;

    // fp16 values passed through flowtable.
    half spatialScale, pointFive, one;
    spatialScale = *(__ubuf__ half *)((__ubuf__ uint16_t *)(&(flowTable->scales[0])));
    pointFive = *(__ubuf__ half *)((__ubuf__ uint16_t *)(&(flowTable->pointFive[0])));
    one = *(__ubuf__ half *)((__ubuf__ uint16_t *)(&(flowTable->indexArr[1])));

    uint32_t pooledHRec = flowTable->pooledHWRec[0];
    uint32_t pooledWRec = flowTable->pooledHWRec[1];

    uint16_t zero = flowTable->indexArr[0];

    uint64_t roiCalcPosXt = flowTable->roiCalcPosXt;
    uint64_t roiCONVXt = flowTable->roiCONVXt;
    uint64_t roiS32SubXt = flowTable->roiS32SubXt;
    uint64_t roiBinHWDeqXt = flowTable->roiBinHWDeqXt;
    uint64_t roiF16OneRepXt = flowTable->roiF16OneRepXt;
    uint64_t roiCONVFXt = flowTable->roiCONVFXt;
    uint64_t binCmpXt = flowTable->binCmpXt;
    uint64_t clearResultXt = flowTable->clearResultXt;
    uint64_t roiVmaxXt = flowTable->roiVmaxXt;
    uint64_t s32Dup2RepXt = flowTable->s32Dup2RepXt;
    uint64_t s32Dup4RepXt = flowTable->s32Dup4RepXt;

    int64_t loadTimes = 0;
    uint64_t outputXm = 0;
    uint64_t inputXm = 0;
    uint64_t roiLoadXm = 0;
    int64_t roiNum = 0;
    int64_t roiGroupLoops = roiParams->roiGroupLoops;
    int64_t ciLoops = flowTable->ciLoops;

    __ubuf__ __docce_buffer(ubFmBuf, half, (__ubuf__ half *)flowTable->ubFeaturMapAddr);
    __ubuf__ __docce_buffer(ubPosBuf, half, (__ubuf__ half *)flowTable->ubPosAddr);

    uint64_t inputWOffSet = flowTable->inputWOffSet;
    uint64_t outputWOffset = flowTable->outputWOffset;
    uint64_t pooledHOffset = flowTable->pooledHOffset;
    uint64_t pooledWOffset = flowTable->pooledWOffset;

    //loop ALL roi groups, each group has at most 128 rois.
    for (int64_t roiGroupLoop = 0; roiGroupLoop < roiGroupLoops; roiGroupLoop++)
    {
        roiNum = roiParams->roiNum;
        roiLoadXm = roiParams->roiLoadXm;
        if (roiGroupLoop == roiGroupLoops - 1)
        {
            roiNum = roiParams->roiNumL;
            roiLoadXm = roiParams->roiLoadXmL;
        }

        //process rois.
        __docce_pipeline
        {
            __docce_stage(loadRois, __docce_wr(ubFmBuf), )
            {
                //roiLoadXm = CalcXmForMov(burstNum, burstLen, srcStride, dstStride); burstNum=1 burstLen=128 srcStride=0 dstStride=0
                //ROI_OFFSET_EACH_LOOP=2048
                copy_gm_to_ubuf(__docce_ptr(ubFmBuf), roiAddr + roiGroupLoop * ROI_OFFSET_EACH_LOOP, roiLoadXm);
            }

            __docce_stage(processRoi, __docce_wr(ubPosBuf), ubFmBuf)
            {
                processRois(__docce_ptr(ubFmBuf), __docce_ptr(ubPosBuf),
                            spatialScale, pointFive, one, pooledHRec, pooledWRec,
                            roiNCHWConvXt, roiCalcPosXt, roiCONVXt, roiS32SubXt, roiBinHWDeqXt, s32Dup4RepXt, roiF16OneRepXt);
                processBins(__docce_ptr(ubFmBuf), __docce_ptr(ubPosBuf), flowTable->ubHiLoopBufAddr[0], flowTable->ubFeaturMapAddr,
                            inputH, inputW, pooledH, pooledW, (__ubuf__ half *)((__ubuf__ uint16_t *)&flowTable->indexArr),
                            roiF16OneRepXt, roiCONVFXt, s32Dup2RepXt, binCmpXt);
            }
        }

        __docce_pipeline
        {
            //loop input channels, each loop process at most 128 channels for input
            for (int64_t ciLoop = 0; ciLoop < ciLoops; ciLoop++)
            {
                loadTimes = flowTable->loadTimes;
                outputXm = flowTable->outputXm;
                inputXm = flowTable->inputXm;

                __docce_stage(loadInput, __docce_wr(ubFmBuf), )
                {
                    if (ciLoop == ciLoops - 1)
                    {
                        loadTimes = flowTable->loadTimesL;
                    }
                    loadInput(loadTimes, inputAddr + ciLoop * inputH * inputW * INDEX_OFFSET,
                              __docce_ptr(ubFmBuf),
                              flowTable->inputOffset, inputXm);
                }

                // must put buff define here
                __ubuf__ __docce_buffer2(ubResultBuf, half, (__ubuf__ half *)flowTable->ubResultAddr[0], (__ubuf__ half *)flowTable->ubResultAddr[1]);

                for (int64_t roi = 0; roi < roiNum; roi++)
                {
                    int64_t hiBufIdx = (roi & 1);
                    __docce_stage(pooling, __docce_wr(ubResultBuf), ubFmBuf)
                    {
                        pooling(__docce_ptr(ubResultBuf), (__ubuf__ uint32_t *)flowTable->ubPosAddr + roi, (__ubuf__ half *)flowTable->ubHiLoopBufAddr[hiBufIdx],
                                pooledH, pooledW,
                                zero, clearResultXt, roiVmaxXt,
                                inputWOffSet, outputWOffset, pooledHOffset, pooledWOffset);
                    }
                    __docce_stage(resultToOut, , ubResultBuf)
                    {
                        if (ciLoop == ciLoops - 1)
                        {
                            outputXm = flowTable->outputXmL;
                        }
                        __gm__ half *outputAddrCurr = outputAddr + roiGroupLoop * roiGroupOffset + roi * roiOutOffset + ciLoop * outCiOffset;

                        copy_ubuf_to_gm(outputAddrCurr, __docce_ptr(ubResultBuf), outputXm);
                        set_flag(PIPE_MTE3, PIPE_S, (event_t)hiBufIdx);
                        wait_flag(PIPE_MTE3, PIPE_S, (event_t)hiBufIdx);
                    }
                    __docce_next(ubResultBuf);
                }
            }
        }
    }
}

inline __aicore__ void loadInput(int64_t loadTimes, __gm__ half *inputAddr, __ubuf__ half *ubFeaturMapAddr, uint64_t addrOffset, uint64_t inputXm) PIPE_ID(PIPE_MTE2)
{
    for (int64_t block = 0; block < loadTimes; block++)
    {
        copy_gm_to_ubuf(ubFeaturMapAddr + block * C0SIZE, inputAddr + block * addrOffset, inputXm);
    }
}

inline PIPE_ID(PIPE_V) __aicore__ void pooling(__ubuf__ half *ubResultBuf,
                                               __ubuf__ uint32_t *ubPosBuf,
                                               __ubuf__ half *ubHiBuf,
                                               int64_t pooledH,
                                               int64_t pooledW,
                                               uint16_t zero,
                                               uint64_t clearResultXt,
                                               uint64_t roiVmaxXt,
                                               uint64_t inputWOffSet,
                                               uint64_t outputWOffset,
                                               uint64_t pooledHOffset,
                                               uint64_t pooledWOffset)
{
    uint64_t hiBufClearXt = CalcXtForOneSrcVectorOP(1, 0, 8, 0, 0);
    uint64_t hiVmaxXt = CalcXtForTwoSrcVectorOP(1, 1, 1, 8, 8, 8, 0);

    vector_dup((__ubuf__ uint16_t *)ubResultBuf, zero, clearResultXt);

    __ubuf__ uint32_t *roiW = ubPosBuf;
    __ubuf__ uint32_t *hAddrBase = roiW + INDEX_OFFSET;
    __ubuf__ uint32_t *wAddrBase = hAddrBase + pooledHOffset;
    __ubuf__ uint32_t *hRepeatBase = wAddrBase + pooledWOffset;
    __ubuf__ uint32_t *wRepeatBase = hRepeatBase + pooledHOffset;

#ifdef __CCE_KT_TEST__
    int32_t currRoiW = 0;
    dvcDebugReadUB((void *)&(currRoiW), (uint64_t)roiW, sizeof(int32_t));
    hiBufClearXt = hiBufClearXt | (((uint64_t)currRoiW) << 56);
    hiVmaxXt = hiVmaxXt | (((uint64_t)currRoiW) << 56);
#else
    uint32_t roiWUb = *roiW;
    hiBufClearXt = hiBufClearXt | ((uint64_t)roiWUb << 56);
    hiVmaxXt = hiVmaxXt | ((uint64_t)roiWUb << 56);
#endif

    __ubuf__ uint32_t *hAddrUB = hAddrBase;

    __ubuf__ uint32_t *hRepeatTimesUB = hRepeatBase;

    __ubuf__ half *resultAddrBufHLoop = ubResultBuf;

    __ubuf__ half *resultAddrBufWLoop;
    __ubuf__ uint32_t *wAddrUB;
    __ubuf__ uint32_t *wRepeatTimesUB;

    for (int64_t ph = 0; ph < pooledH; ph++)
    {
        vector_dup((__ubuf__ uint16_t *)ubHiBuf, zero, hiBufClearXt);
        pipe_barrier(PIPE_V);

#ifdef __CCE_KT_TEST__
        uint32_t hAddr = 0;
        uint32_t hRepeatTimes = 0;
        dvcDebugReadUB((void *)&(hAddr), (uint64_t)hAddrUB, sizeof(uint32_t));
        dvcDebugReadUB((void *)&(hRepeatTimes), (uint64_t)hRepeatTimesUB, sizeof(uint32_t));

        for (int64_t hi = 0; hi < hRepeatTimes; hi++)
        {
            vmax((__ubuf__ half *)ubHiBuf, (__ubuf__ half *)hAddr + hi * inputWOffSet, (__ubuf__ half *)ubHiBuf, hiVmaxXt);
            pipe_barrier(PIPE_V);
        }
#else
        __ubuf__ half *hiOffset = (__ubuf__ half *)(uint64_t)*hAddrUB;
        for (int64_t hi = 0; hi < *hRepeatTimesUB; hi++)
        {
            vmax((__ubuf__ half *)ubHiBuf, hiOffset, (__ubuf__ half *)ubHiBuf, hiVmaxXt);
            pipe_barrier(PIPE_V);
            hiOffset += inputWOffSet;
        }
#endif
        resultAddrBufWLoop = resultAddrBufHLoop;
        wAddrUB = wAddrBase;
        wRepeatTimesUB = wRepeatBase;
        for (int64_t pw = 0; pw < pooledW; pw++)
        {

#ifdef __CCE_KT_TEST__
            uint32_t wAddr = 0;
            uint32_t wRepeatTimes = 0;
            dvcDebugReadUB((void *)&(wAddr), (uint64_t)wAddrUB, sizeof(uint32_t));
            dvcDebugReadUB((void *)&(wRepeatTimes), (uint64_t)wRepeatTimesUB, sizeof(uint32_t));

            uint64_t roiVmaxXtWithRepat = roiVmaxXt | ((uint64_t)wRepeatTimes << 56);
            vmax(resultAddrBufWLoop, (__ubuf__ half *)wAddr, resultAddrBufWLoop, roiVmaxXtWithRepat);
#else
            uint64_t roiVmaxXtWithRepat = roiVmaxXt | ((uint64_t)*wRepeatTimesUB << 56);
            vmax(resultAddrBufWLoop, (__ubuf__ half *)(uint64_t)*wAddrUB, resultAddrBufWLoop, roiVmaxXtWithRepat);
#endif
            resultAddrBufWLoop += 16;
            wAddrUB += INDEX_OFFSET;
            wRepeatTimesUB += INDEX_OFFSET;
        }
        resultAddrBufHLoop += outputWOffset;
        hAddrUB += INDEX_OFFSET;
        hRepeatTimesUB += INDEX_OFFSET;
    }
}

inline PIPE_ID(PIPE_V) __aicore__ void processRois(__ubuf__ half *ubBufA,
                                                   __ubuf__ half *ubBufB,
                                                   half spatialScale,
                                                   half pointFive,
                                                   half one,
                                                   uint32_t pooledHRec,
                                                   uint32_t pooledWRec,
                                                   uint64_t roiNCHWConvXt,
                                                   uint64_t roiCalcPosXt,
                                                   uint64_t roiCONVXt,
                                                   uint64_t roiS32SubXt,
                                                   uint64_t roiBinHWDeqXt,
                                                   uint64_t s32Dup4RepXt,
                                                   uint64_t roiF16OneRepXt)
{
    //BLOCK_NUM=8
    //BLOCK_SIZE=32
    uint64_t src1Addrs[BLOCK_NUM], src2Addrs[BLOCK_NUM], dst1Addrs[BLOCK_NUM], dst2Addrs[BLOCK_NUM];
    for (int64_t i = 0; i < BLOCK_NUM; i++)
    {
        src1Addrs[i] = (uint64_t)ubBufA + i * BLOCK_SIZE;
        src2Addrs[i] = (uint64_t)ubBufA + (i + BLOCK_NUM) * BLOCK_SIZE;
        dst1Addrs[i] = (uint64_t)ubBufB + i * BLOCK_SIZE * BLOCK_NUM;
        dst2Addrs[i] = (uint64_t)ubBufB + (i + BLOCK_NUM) * BLOCK_SIZE * BLOCK_NUM;
    }
    set_va_reg_sb(VA0, src1Addrs);
    set_va_reg_sb(VA1, src2Addrs);
    set_va_reg_sb(VA2, dst1Addrs);
    set_va_reg_sb(VA3, dst2Addrs);
    //roiNCHWConvXt = CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, 2 * BLOCKNUM, STRIDE_ZERO, STRIDE_ZERO, BLOCKNUM);
    //DEFAULT_VECTOR_XD_STRIDE=1,    BLOCKNUM=8, STRIDE_ZERO=0
    scatter_vnchwconv_b16(VA2, VA0, roiNCHWConvXt);
    pipe_barrier(PIPE_V);

    // rois(sh eh sw ew) * spatialScale
    //CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, DEFAULT_VECTOR_XN_STRIDE, DEFAULT_VECTOR_XD_REPEAT, DEFAULT_VECTOR_XN_REPEAT, DATA_COUNT_OF_EACH_ROI);
    //DEFAULT_VECTOR_XD_STRIDE=1, DEFAULT_VECTOR_XN_STRIDE=1, DEFAULT_VECTOR_XD_REPEAT=8, DEFAULT_VECTOR_XN_REPEAT=8, DATA_COUNT_OF_EACH_ROI=4
    vmuls(ubBufA, ubBufB + INDEX_OFFSET, spatialScale, roiCalcPosXt);//INDEX_OFFSET=128
    pipe_barrier(PIPE_V);

    vadds(ubBufB, ubBufA, pointFive, roiCalcPosXt);
    pipe_barrier(PIPE_V);

    //conv float16 to int floor, int32_t at BufA. roiStartW, roiStartH, roiEndW, roiEndH
    //CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, DEFAULT_VECTOR_XN_STRIDE, DEFAULT_VECTOR_XD_REPEAT, DEFAULT_VECTOR_XN_REPEAT / sizeof(uint16_t), BLOCKNUM);
    //DEFAULT_VECTOR_XD_STRIDE=1, DEFAULT_VECTOR_XN_STRIDE=1, DEFAULT_VECTOR_XD_REPEAT=8, DEFAULT_VECTOR_XN_REPEAT=8, BLOCKNUM=8
    vconv_f162s32f((__ubuf__ int32_t *)ubBufA, ubBufB, roiCONVXt);
    pipe_barrier(PIPE_V);

    // BINHW x2-x1, y2-y1
    //roiS32SubXt = CalcXtForTwoSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE=1,
    // DEFAULT_VECTOR_XN_STRIDE=1, DEFAULT_VECTOR_XM_STRIDE=1, DEFAULT_VECTOR_XD_REPEAT=8, DEFAULT_VECTOR_XN_REPEAT=8, DEFAULT_VECTOR_XM_REPEAT=8, 2 * ROI_EACH_LOOP=128 / INT32_NUM_EACH_REPEAT=64);
    vsub((__ubuf__ int32_t *)ubBufB, (__ubuf__ int32_t *)ubBufA + 2 * INDEX_OFFSET, (__ubuf__ int32_t *)ubBufA, roiS32SubXt);
    pipe_barrier(PIPE_V);

    vector_dup((__ubuf__ uint32_t *)ubBufA + 4 * INDEX_OFFSET, (uint32_t)1, s32Dup4RepXt);
    pipe_barrier(PIPE_V);

    //@@@@@@@@@ ubPosBuf keep: x2 - x1 +1 ; y2 - y1 + 1 int32
    vadd((__ubuf__ int32_t *)ubBufB, (__ubuf__ int32_t *)ubBufA + 4 * INDEX_OFFSET, (__ubuf__ int32_t *)ubBufB, roiS32SubXt);
    pipe_barrier(PIPE_V);

    vmax((__ubuf__ int32_t *)ubBufB, (__ubuf__ int32_t *)ubBufA + 4 * INDEX_OFFSET, (__ubuf__ int32_t *)ubBufB, roiS32SubXt);
    pipe_barrier(PIPE_V);

    // roi hw int32_t -> fp16, fp16 at BufA + 64KB
    //roiBinHWDeqXt = CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, DEFAULT_VECTOR_XN_STRIDE, DEFAULT_VECTOR_XD_REPEAT / sizeof(uint16_t), DEFAULT_VECTOR_XN_REPEAT, 2 * 2 * DEFAULT_REPEAT_TIME);
    //DEFAULT_VECTOR_XD_STRIDE=1,DEFAULT_VECTOR_XN_STRIDE=1,DEFAUT_REPEAT_TIME=1
    __ubuf__ half *binAddrStart = ubBufA + 32 * 1024;
    vconv_deq(binAddrStart, (__ubuf__ int32_t *)ubBufB, roiBinHWDeqXt);
    pipe_barrier(PIPE_V);

    vector_dup((__ubuf__ uint32_t *)binAddrStart + 4 * INDEX_OFFSET, pooledHRec, s32Dup4RepXt);
    pipe_barrier(PIPE_V);

    vector_dup((__ubuf__ uint32_t *)binAddrStart + 5 * INDEX_OFFSET, pooledWRec, s32Dup4RepXt);
    pipe_barrier(PIPE_V);

    vconv_f162f32((__ubuf__ float *)binAddrStart + 6 * INDEX_OFFSET, binAddrStart, roiCONVXt);
    pipe_barrier(PIPE_V);

    vmul((__ubuf__ float *)binAddrStart + 8 * INDEX_OFFSET, (__ubuf__ float *)binAddrStart + 6 * INDEX_OFFSET, (__ubuf__ float *)binAddrStart + 4 * INDEX_OFFSET, roiS32SubXt);
    pipe_barrier(PIPE_V);

    vconv_f322f16(binAddrStart, (__ubuf__ float *)binAddrStart + 8 * INDEX_OFFSET, roiBinHWDeqXt);
    pipe_barrier(PIPE_V);

    // //@@@@@@@@@ ubPosBuf keep, binW offset: 2*128*int32
    // vmuls(binAddrStart, binAddrStart, pooledWRecip, roiF16OneRepXt);
    // pipe_barrier(PIPE_V);

    // //@@@@@@@@@ ubPosBuf keep: binH offset: 2*128*int32 + 128*fp16
    // vmuls(binAddrStart + INDEX_OFFSET, binAddrStart + INDEX_OFFSET, pooledHRecip, roiF16OneRepXt);
    // pipe_barrier(PIPE_V);
}

inline PIPE_ID(PIPE_V) __aicore__ void processBins(__ubuf__ half *roiS32Addr,
                                                   __ubuf__ half *ubPosBuf,
                                                   uint64_t poolBaseAddrVal,
                                                   uint64_t fmAddrVal,
                                                   int64_t inputH,
                                                   int64_t inputW,
                                                   int64_t pooledH,
                                                   int64_t pooledW,
                                                   __ubuf__ half *indexAddr,
                                                   uint64_t roiF16OneRepXt,
                                                   uint64_t roiCONVFXt,
                                                   uint64_t s32Dup2RepXt,
                                                   uint64_t binCmpXt)
{
    __ubuf__ int32_t *xStart = (__ubuf__ int32_t *)roiS32Addr;
    __ubuf__ int32_t *yStart = xStart + INDEX_OFFSET;
    __ubuf__ int32_t *xEnd = yStart + INDEX_OFFSET;
    __ubuf__ int32_t *yEnd = xEnd + INDEX_OFFSET;

    /******
     * START AT 64KB
     * fp16         BINW |  128 * 2B
     * fp16         tmp0 |  128 * 2B  wstart = bin_w * pw // wend = bin_w * (pw + 1)
     * fp16         tmp1 |  128 * 2B  hstart = bin_h * ph // hend = bin_h * (ph + 1)
     * fp32 floor wstart |  128 *4B   floor(wstart)
     *                   |
     * fp32 ceil  hstart |  128 *4B   floor(hstart)
     *                   |
     * fp32 ceil  wend   |  128 *4B   ceil(wend)
     *                   |
     * fp32 ceil  hend   |  128 *4B   ceil(hend)
     *                   |
     * fp32         tmp2 | 
     * fp32         tmp3 | zeros 128 * 4B
     * fp32         tmpH | H    128 * 4B
     * fp32         tmpW | W    128 * 4B
     * fp32 hoffSetSizeAddrS32 | 128 * 4B
     * fp32 woffSetSizeAddrS32 | 128 * 4B
     * fp32 poolBaseAddr | 128 * 4B
    */

    __ubuf__ half *binWAddr = roiS32Addr + 32 * 1024;
    __ubuf__ half *binHAddr = binWAddr + INDEX_OFFSET;
    __ubuf__ half *temp0Addr = binHAddr + INDEX_OFFSET;
    __ubuf__ half *temp1Addr = temp0Addr + INDEX_OFFSET;

    __ubuf__ int32_t *wstart = (__ubuf__ int32_t *)(temp1Addr + INDEX_OFFSET);
    __ubuf__ int32_t *hstart = wstart + INDEX_OFFSET;
    __ubuf__ int32_t *wend = hstart + INDEX_OFFSET;
    __ubuf__ int32_t *hend = wend + INDEX_OFFSET;

    __ubuf__ int32_t *tmp2AddrS32 = hend + INDEX_OFFSET;
    __ubuf__ int32_t *tmp3AddrS32 = tmp2AddrS32 + INDEX_OFFSET;

    __ubuf__ int32_t *tmpHAddrS32 = tmp3AddrS32 + INDEX_OFFSET;
    __ubuf__ int32_t *tmpWAddrS32 = tmpHAddrS32 + INDEX_OFFSET;

    __ubuf__ int32_t *hiOffSetAddrS32 = tmpWAddrS32 + INDEX_OFFSET;
    __ubuf__ int32_t *woffSetSizeAddrS32 = hiOffSetAddrS32 + INDEX_OFFSET;

    __ubuf__ int32_t *poolBaseAddrS32 = woffSetSizeAddrS32 + INDEX_OFFSET;

    __ubuf__ int32_t *fmBaseAddrS32 = poolBaseAddrS32 + INDEX_OFFSET;

    //POS RESULT ADDRRS, AFTER REGION H AND W
    __ubuf__ int32_t *hAddrs = (__ubuf__ int32_t *)ubPosBuf + INDEX_OFFSET;
    __ubuf__ int32_t *wAddrs = hAddrs + pooledH * INDEX_OFFSET;
    __ubuf__ int32_t *hRepeats = wAddrs + pooledW * INDEX_OFFSET;
    __ubuf__ int32_t *wRepeats = hRepeats + pooledH * INDEX_OFFSET;

    // w0
    __ubuf__ int32_t *w0Start = wRepeats + pooledW * INDEX_OFFSET;

    //tem3 set zeros.
    vector_dup((__ubuf__ uint32_t *)tmp3AddrS32, (uint32_t)0, s32Dup2RepXt);

    //temH   all inputH
    vector_dup((__ubuf__ uint32_t *)tmpHAddrS32, inputH, s32Dup2RepXt);

    //temW   all inputW
    vector_dup((__ubuf__ uint32_t *)tmpWAddrS32, inputW, s32Dup2RepXt);

    // pingpong buf for Hi loop offset, 24KB
    vector_dup((__ubuf__ uint32_t *)hiOffSetAddrS32, (uint32_t)(24 * 1024), s32Dup2RepXt);

    // 128 * sizeof(fp16)
    vector_dup((__ubuf__ uint32_t *)woffSetSizeAddrS32, INDEX_OFFSET * sizeof(uint16_t), s32Dup2RepXt);
    pipe_barrier(PIPE_V);

    // poolBaseAddr
    vector_dup((__ubuf__ uint32_t *)poolBaseAddrS32, (uint32_t)poolBaseAddrVal, s32Dup2RepXt);
    pipe_barrier(PIPE_V);

    // fmBaseAddr
    vector_dup((__ubuf__ uint32_t *)fmBaseAddrS32, (uint32_t)fmAddrVal, s32Dup2RepXt);
    pipe_barrier(PIPE_V);

    vmin((__ubuf__ int32_t *)ubPosBuf, (__ubuf__ int32_t *)ubPosBuf, tmpWAddrS32, binCmpXt);
    pipe_barrier(PIPE_V);

    int64_t pw = 0;
    for (int64_t ph = 0; ph < pooledH; ph++)
    {
        half indexPh = indexAddr[ph];
        half indexPw = indexAddr[pw];

        half indexPhPlus1 = indexAddr[ph + 1];
        half indexPwPlus1 = indexAddr[pw + 1];

        // store to tmp0
        vmuls(temp0Addr, binWAddr, indexPw, roiF16OneRepXt); //wstart = bin_w * pw
        pipe_barrier(PIPE_V);

        // store to tmp1
        vmuls(temp1Addr, binHAddr, indexPh, roiF16OneRepXt); //hstart = bin_h * ph
        pipe_barrier(PIPE_V);

        // conv float16 to int floor.
        vconv_f162s32f(wstart, temp0Addr, roiCONVFXt);
        pipe_barrier(PIPE_V);

        vmuls(temp0Addr, binWAddr, indexPwPlus1, roiF16OneRepXt); // wend = bin_w * (pw + 1)
        pipe_barrier(PIPE_V);

        vmuls(temp1Addr, binHAddr, indexPhPlus1, roiF16OneRepXt); // hend = bin_h * (ph + 1)
        pipe_barrier(PIPE_V);

        // conv float16 to int floor.
        vconv_f162s32c(wend, temp0Addr, roiCONVFXt);
        pipe_barrier(PIPE_V);

        // wstart + roi_start_w
        vadd(tmp2AddrS32, wstart, xStart, binCmpXt);
        pipe_barrier(PIPE_V);
        //max((wstart + roi_start_w), 0)
        vmax(tmp2AddrS32, tmp2AddrS32, tmp3AddrS32, binCmpXt);
        pipe_barrier(PIPE_V);
        //min(max(wstart + roi_start_w, 0), width_) = wstart
        vmin(wstart, tmp2AddrS32, tmpWAddrS32, binCmpXt);
        pipe_barrier(PIPE_V);

        // hstart + roi_start_h
        vadd(tmp2AddrS32, hstart, yStart, binCmpXt);
        pipe_barrier(PIPE_V);
        //max((hstart + roi_start_h), 0)
        vmax(tmp2AddrS32, tmp2AddrS32, tmp3AddrS32, binCmpXt);
        pipe_barrier(PIPE_V);
        //min(max(hstart + roi_start_h, 0), height_) = hstart
        vmin(hstart, tmp2AddrS32, tmpHAddrS32, binCmpXt);
        pipe_barrier(PIPE_V);

        // hend + roi_start_h
        vadd(tmp2AddrS32, hend, yStart, binCmpXt);
        pipe_barrier(PIPE_V);
        //max((hend + roi_start_h), 0)
        vmax(tmp2AddrS32, tmp2AddrS32, tmp3AddrS32, binCmpXt);
        pipe_barrier(PIPE_V);
        //min(max(hend + roi_start_h, 0), height_) = hend
        vmin(hend, tmp2AddrS32, tmpHAddrS32, binCmpXt);
        pipe_barrier(PIPE_V);

        // wend + roi_start_w
        vadd(tmp2AddrS32, wend, xStart, binCmpXt);
        pipe_barrier(PIPE_V);
        //max((wend + roi_start_w), 0)
        vmax(tmp2AddrS32, tmp2AddrS32, tmp3AddrS32, binCmpXt);
        pipe_barrier(PIPE_V);
        //min(max(wend + roi_start_w, 0), width_) = wend
        vmin(wend, tmp2AddrS32, tmpWAddrS32, binCmpXt);
        pipe_barrier(PIPE_V);

        /******
         * ubPosBuf
         *
         * int32      x2-x1+1|  128 * 4B
         * int32      y2-y1+1|  128 * 4B
         * int32      H addrs|  128 * 2B
         * int32      W addrs|  128 * 2B
         * int32     H repeat|  128 * 2B
         * int32     W repeat|  128 * 2B
         * 
         */

        //now we have wstart, hstart, hend, wend
        // h * inputW
        vmul(hAddrs + ph * INDEX_OFFSET, tmpWAddrS32, hstart, binCmpXt);
        pipe_barrier(PIPE_V);

        // h * inputW + w
        vadd(hAddrs + ph * INDEX_OFFSET, hAddrs + ph * INDEX_OFFSET, wstart, binCmpXt);
        pipe_barrier(PIPE_V);

        // (h * inputW + w) * 128 * sizeof(fp16)
        vmul(hAddrs + ph * INDEX_OFFSET, woffSetSizeAddrS32, hAddrs + ph * INDEX_OFFSET, binCmpXt);
        pipe_barrier(PIPE_V);

        // (h * inputW + w) * 128 * sizeof(fp16) + fmAddr = HAddr
        vadd(hAddrs + ph * INDEX_OFFSET, fmBaseAddrS32, hAddrs + ph * INDEX_OFFSET, binCmpXt);
        pipe_barrier(PIPE_V);

        //repeat H direction
        vsub(hRepeats + ph * INDEX_OFFSET, hend, hstart, binCmpXt);
        pipe_barrier(PIPE_V);
    }

    int64_t ph = 0;
    for (int64_t pw = 0; pw < pooledW; pw++)
    {
        half indexPh = indexAddr[ph];
        half indexPw = indexAddr[pw];

        half indexPhPlus1 = indexAddr[ph + 1];
        half indexPwPlus1 = indexAddr[pw + 1];

        // store to tmp0
        vmuls(temp0Addr, binWAddr, indexPw, roiF16OneRepXt); //wstart = bin_w * pw
        pipe_barrier(PIPE_V);
        // store to tmp1
        vmuls(temp1Addr, binHAddr, indexPh, roiF16OneRepXt); //hstart = bin_h * ph
        pipe_barrier(PIPE_V);

        // conv float16 to int floor.
        vconv_f162s32f(wstart, temp0Addr, roiCONVFXt);
        pipe_barrier(PIPE_V);

        vmuls(temp0Addr, binWAddr, indexPwPlus1, roiF16OneRepXt); // wend = bin_w * (pw + 1)
        pipe_barrier(PIPE_V);

        vmuls(temp1Addr, binHAddr, indexPhPlus1, roiF16OneRepXt); // hend = bin_h * (ph + 1)
        pipe_barrier(PIPE_V);

        // conv float16 to int floor.
        vconv_f162s32c(wend, temp0Addr, roiCONVFXt);
        pipe_barrier(PIPE_V);

        // wstart + roi_start_w
        vadd(tmp2AddrS32, wstart, xStart, binCmpXt);
        pipe_barrier(PIPE_V);

        //max((wstart + roi_start_w), 0)
        vmax(tmp2AddrS32, tmp2AddrS32, tmp3AddrS32, binCmpXt);
        pipe_barrier(PIPE_V);

        //min(max(wstart + roi_start_w, 0), width_) = wstart
        vmin(wstart, tmp2AddrS32, tmpWAddrS32, binCmpXt);
        pipe_barrier(PIPE_V);

        if (0 == pw)
        {
            vmin(w0Start, tmp2AddrS32, tmpWAddrS32, binCmpXt);
            pipe_barrier(PIPE_V);
        }

        // hstart + roi_start_h
        vadd(tmp2AddrS32, hstart, yStart, binCmpXt);
        pipe_barrier(PIPE_V);
        //max((hstart + roi_start_h), 0)
        vmax(tmp2AddrS32, tmp2AddrS32, tmp3AddrS32, binCmpXt);
        pipe_barrier(PIPE_V);
        //min(max(hstart + roi_start_h, 0), height_) = hstart
        vmin(hstart, tmp2AddrS32, tmpHAddrS32, binCmpXt);
        pipe_barrier(PIPE_V);

        // hend + roi_start_h
        vadd(tmp2AddrS32, hend, yStart, binCmpXt);
        pipe_barrier(PIPE_V);
        //max((hend + roi_start_h), 0)
        vmax(tmp2AddrS32, tmp2AddrS32, tmp3AddrS32, binCmpXt);
        pipe_barrier(PIPE_V);
        //min(max(hend + roi_start_h, 0), height_) = hend
        vmin(hend, tmp2AddrS32, tmpHAddrS32, binCmpXt);
        pipe_barrier(PIPE_V);

        // wend + roi_start_w
        vadd(tmp2AddrS32, wend, xStart, binCmpXt);
        pipe_barrier(PIPE_V);
        //max((wend + roi_start_w), 0)
        vmax(tmp2AddrS32, tmp2AddrS32, tmp3AddrS32, binCmpXt);
        pipe_barrier(PIPE_V);
        //min(max(wend + roi_start_w, 0), width_) = wend
        vmin(wend, tmp2AddrS32, tmpWAddrS32, binCmpXt);
        pipe_barrier(PIPE_V);

        /******
         * ubPosBuf
         *
         * int32      x2-x1+1|  128 * 4B
         * int32      y2-y1+1|  128 * 4B
         * int32      H addrs|  128 * 2B
         * int32      W addrs|  128 * 2B
         * int32     H repeat|  128 * 2B
         * int32     W repeat|  128 * 2B
         * 
         */

        //now we have wstart, hstart, hend, wend
        vsub(wAddrs + pw * INDEX_OFFSET, wstart, w0Start, binCmpXt);
        pipe_barrier(PIPE_V);

        vmul(wAddrs + pw * INDEX_OFFSET, wAddrs + pw * INDEX_OFFSET, woffSetSizeAddrS32, binCmpXt);
        pipe_barrier(PIPE_V);

        // add Hi base
        vadd(wAddrs + pw * INDEX_OFFSET, wAddrs + pw * INDEX_OFFSET, poolBaseAddrS32, binCmpXt);
        pipe_barrier(PIPE_V);

        set_vector_mask((uint64_t)0xAAAAAAAAAAAAAAAA, (uint64_t)0xAAAAAAAAAAAAAAAA);
        vadd(wAddrs + pw * INDEX_OFFSET, wAddrs + pw * INDEX_OFFSET, hiOffSetAddrS32, binCmpXt);
        pipe_barrier(PIPE_V);

        set_vector_mask(-1, -1);
        //repeat H direction
        vsub(wRepeats + pw * INDEX_OFFSET, wend, wstart, binCmpXt);
        pipe_barrier(PIPE_V);
    }
}
}; // namespace kernel
}; // namespace cce

#endif

```
