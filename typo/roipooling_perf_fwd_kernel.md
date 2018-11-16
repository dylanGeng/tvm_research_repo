```
#include "catch.hpp"
#include "env.hpp"
#include "cce_middef.hpp"
#include "cce_opiddef.hpp"
#include "cce_errcodedef.hpp"
#include "cce_util.hpp"
#include "fp16_t.hpp"
#include "compiler_stub.hpp"
#include "kernel_creatorreg.hpp"
#include "roipooling_perf_flowtable.hpp"
#include "calc_instr_reg.hpp"
#include "roipooling_perf_fwd_kernel.hpp"
#include "dnn_comm.hpp"

#ifdef __CCE_KT_TEST__
#include "../../llt/cce/ut/testframe/utils/remote_core.hpp"
#endif
#ifdef __CCE_UT_TEST__
#include "../../llt/cce/ut/testframe/utils/remote_core.hpp"
#endif
#include <math.h>
using namespace std;

namespace cce
{
namespace kernel
{
extern const char *g_l2fusion;
extern void printL2RemaptblInfo(rtL2Ctrl_t &l2ctrl);
RoipoolingPerfForwardKernel::RoipoolingPerfForwardKernel()
{
    memset_s(&ubPara, sizeof(kRoipoolingUbPara), 0x00, sizeof(ubPara));
    memset_s(&l2Para, sizeof(kRoipoolingL2Para), 0x00, sizeof(l2Para));
    memset_s(&taskL2Info, sizeof(TaskROIPoolingL2Info), 0x00, sizeof(TaskROIPoolingL2Info));
    taskL2Info.inputL2AddrInfo.l2DataIndex = CC_L2_MAXDATANUM;
    taskL2Info.roisL2AddrInfo.l2DataIndex = CC_L2_MAXDATANUM;
    taskL2Info.outputL2AddrInfo.l2DataIndex = CC_L2_MAXDATANUM;
    SetPlatformParam();
}
RoipoolingPerfForwardKernel::~RoipoolingPerfForwardKernel()
{
}

uint32_t RoipoolingPerfForwardKernel::SaveL2Para(AttrList &attrList, AttrList &dataAddrList)
{
    uint32_t attrLen = 0;
    const void *attrValue;

    L2AddrInfo *ptmpL2AddrInfo = NULL;

    attrList.Get(ROIPOOLING_KERNEL_TENSOR_INPUT, attrLen, attrValue);
    kTensor_t *inputTensor = (kTensor_t *)attrValue;

    attrList.Get(ROIPOOLING_KERNEL_TENSOR_OUTPUT, attrLen, attrValue);
    kTensor_t *outputTensor = (kTensor_t *)attrValue;

    attrList.Get(ROIPOOLING_KERNEL_PARAM, attrLen, attrValue);
    kRoipoolingParam_t *prarm = (kRoipoolingParam_t *)attrValue;

    dataAddrList.Get(ROIPOOLING_KERNEL_DATA_INPUT, attrLen, attrValue);
    uint64_t inputAddr = *((uint64_t *)attrValue);

    dataAddrList.Get(ROIPOOLING_KERNEL_ROIS_INPUT, attrLen, attrValue);
    uint64_t roisAddr = *((uint64_t *)attrValue);

    dataAddrList.Get(ROIPOOLING_KERNEL_DATA_OUTPUT, attrLen, attrValue);
    uint64_t outputAddr = *((uint64_t *)attrValue);

    //task max pagenum
    attrValue = NULL;
    attrList.Get(ROIPOOLING_TASK_PARA_L2_MAXPAGENUM, attrLen, attrValue);
    if (NULL != attrValue)
    {
#ifndef __CCE_KT_TEST__
        taskL2Info.maxPageNum = *(uint64_t *)attrValue;
#endif
    }
    //input
    attrValue = NULL;
    attrList.Get(ROIPOOLING_INPUT_L2_ADDR_INFO, attrLen, attrValue);
    ptmpL2AddrInfo = (L2AddrInfo *)attrValue;
    if (NULL != attrValue && ptmpL2AddrInfo->l2DataIndex < CC_L2_MAXDATANUM)
        taskL2Info.inputL2AddrInfo = *(L2AddrInfo *)attrValue;
    //rois
    attrValue = NULL;
    attrList.Get(ROIPOOLING_ROIS_L2_ADDR_INFO, attrLen, attrValue);
    ptmpL2AddrInfo = (L2AddrInfo *)attrValue;
    if (NULL != attrValue && ptmpL2AddrInfo->l2DataIndex < CC_L2_MAXDATANUM)
        taskL2Info.roisL2AddrInfo = *(L2AddrInfo *)attrValue;

    //output
    attrValue = NULL;
    ptmpL2AddrInfo = NULL;
    attrList.Get(ROIPOOLING_OUTPUT_L2_ADDR_INFO, attrLen, attrValue);
    ptmpL2AddrInfo = (L2AddrInfo *)attrValue;
    if (NULL != attrValue && ptmpL2AddrInfo->l2DataIndex < CC_L2_MAXDATANUM)
        taskL2Info.outputL2AddrInfo = *(L2AddrInfo *)attrValue;
    //remaptbl
    attrValue = NULL;
    attrList.Get(ROIPOOLING_L2_RENAPTBL_INFO, attrLen, attrValue);
    if (NULL != attrValue)
        taskL2Info.remapTbl = *(rtL2Ctrl_t *)attrValue;

    memcpy_s(&l2Para.input, sizeof(kTensor_t), inputTensor, sizeof(kTensor_t));
    memcpy_s(&l2Para.output, sizeof(kTensor_t), outputTensor, sizeof(kTensor_t));
    memcpy_s(&l2Para.roiParm, sizeof(kRoipoolingParam_t), prarm, sizeof(kRoipoolingParam_t));

    l2Para.inputAddr = inputAddr;
    l2Para.outputAddr = outputAddr;
    l2Para.roiAddr = roisAddr;
    return CCE_SUCCESS;
}

uint32_t RoipoolingPerfForwardKernel::SaveUbPara()
{
    /*Using special UB params, every scene use its own UB params.*/

    //ubPara.ubInputAddr[0] = SCEN_ONEC0_POS_ADDR; //unused in this flowtable.
    //ubPara.ubInputAddr[1] = ubPara.ubInputAddr[0] + UB_ONE_BUFFER_SIZE;//unused in this flowtable.
    //ubPara.ubOutputAddr[0] = ubPara.ubInputAddr[1] + UB_ONE_BUFFER_SIZE;//unused in this flowtable.
    //ubPara.ubOutputAddr[1] = ubPara.ubOutputAddr[0] + UB_ONE_BUFFER_SIZE;//unused in this flowtable.
    //ubPara.flowtableAddr = ubPara.ubInputAddr[0] + UB_ONE_BUFFER_VALID_SIZE; //unused in this flowtable.
    //ubPara.roiRawDataAddr = ubPara.ubInputAddr[1] + UB_ONE_BUFFER_VALID_SIZE;//unused in this flowtable.
    //ubPara.roiSpatitaledAddr = ubPara.ubOutputAddr[0] + UB_ONE_BUFFER_VALID_SIZE;//unused in this flowtable.

    return CCE_SUCCESS;
}

uint32_t RoipoolingPerfForwardKernel::CalcFlowTableSize(AttrList &attrList, AttrList &dataAddrList, uint32_t &flowTableSize)
{
    /*Clear static check for unused parameters.*/
    (void)attrList;
    (void)dataAddrList;
    flowTableSize = sizeof(kRoipoolingPerfFlowTable_t);
    flowTableSize = CEIL(flowTableSize, CC_UB_ALIGN_SIZE) * CC_UB_ALIGN_SIZE;
    return CCE_SUCCESS;
}

uint32_t RoipoolingPerfForwardKernel::GenFlowTable(AttrList &attrList, AttrList &dataAddrList,
                                                   uint32_t flowTableSize, void *flowTable)
{
    flowTableSize = flowTableSize - 0; //clear static check:unused parameters.
    uint64_t offlineBaseAddr = 0;
    uint64_t onlineBaseAddr = 0;
    if (CCE_SUCCESS == GetTilingPlanAddrOffset(attrList, offlineBaseAddr, onlineBaseAddr))
    {
        kRoipoolingPerfFlowTable_t *table = (kRoipoolingPerfFlowTable_t *)flowTable;
        table->inputAddr = table->inputAddr + onlineBaseAddr - offlineBaseAddr;
        table->outputAddr = table->outputAddr + onlineBaseAddr - offlineBaseAddr;
        table->roiAddr = table->roiAddr + onlineBaseAddr - offlineBaseAddr;
        return CCE_SUCCESS;
    }

#ifdef __CCE_KT_TEST__
    SetPlatformParam();
#endif

    uint32_t ret = CCE_SUCCESS;
    ret = SaveL2Para(attrList, dataAddrList);
    JUDGE_RETURN_VALUE_LOG(ret, "SaveL2Para fail !");

    ret = SaveUbPara();
    JUDGE_RETURN_VALUE_LOG(ret, "SaveUbPara fail !");

    ret = BuildFlowTablePara(flowTable);
    JUDGE_RETURN_VALUE_LOG(ret, "BuildFlowTablePara fail !");

    return CCE_SUCCESS;
}

void RoipoolingPerfForwardKernel::BuildRoiMultiBatchParams(void *flowTable, uint32_t oneBatchRoiNum, uint32_t roiParaIdx)
{ //Each batch could handle 128 rois, the dims -->roiGroupLoops->128->8*C0->1->1
    kRoipoolingPerfFlowTable_t *table = (kRoipoolingPerfFlowTable_t *)flowTable;
    kRoiMulitBatchParams_t *roiParams = &table->roiMultiBatchParams[roiParaIdx];

    roiParams->roiGroupLoops = CEIL((int64_t)oneBatchRoiNum, BLOCKNUM * (int64_t)l2Para.input.channel0); // Mode 128
    uint64_t burstNum, burstLen, srcStride, dstStride;
    burstNum = (uint64_t)DEFULT_MOV_BURSTS;                //1
    burstLen = (uint64_t)BLOCKNUM * l2Para.input.channel0; //128
    srcStride = STRIDE_ZERO;                               //0
    dstStride = STRIDE_ZERO;                               //0
    roiParams->roiNum = ROI_EACH_LOOP;                     //128
    roiParams->roiLoadXm = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
    roiParams->roiNumL = oneBatchRoiNum % ROI_EACH_LOOP;
    burstLen = (uint64_t)(roiParams->roiNumL);
    roiParams->roiLoadXmL = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
#ifndef __CCE_KT_TEST__
    if (0 == roiParams->roiNumL)
    {
        roiParams->roiLoadXmL = roiParams->roiLoadXm;
        roiParams->roiNumL = ROI_EACH_LOOP;
    }
#endif
}

void RoipoolingPerfForwardKernel::SaveCommonParams(void *flowTable)
{
    kRoipoolingPerfFlowTable_t *table = (kRoipoolingPerfFlowTable_t *)flowTable;
    table->inputH = l2Para.input.height;
    table->inputW = l2Para.input.width;

    table->pooledH = l2Para.roiParm.pooledH;
    table->pooledW = l2Para.roiParm.pooledW;

    //multi batch
#ifdef __CCE_KT_TEST__
    table->roiMultiBatchNum = 1;
    // multibatch within input batch
    table->multiBatchOffsetInput = l2Para.input.height * l2Para.input.width * l2Para.input.channel * l2Para.input.channel0 * sizeof(uint16_t);
    table->multiBatchOffsetRoi = 0;
    table->multiBatchOffsetOutput = l2Para.roiParm.riosNum * l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * l2Para.input.channel * l2Para.input.channel0 * sizeof(uint16_t);

    BuildRoiMultiBatchParams(flowTable, l2Para.roiParm.riosNum, 0);
    table->roiMultiBatchParams[1] = table->roiMultiBatchParams[0];
#else
    if (1 != l2Para.input.batchSize)
    {
        CCE_LOG(CC_LOG_INFO, "SaveCommonParams multi-batch multi-core ! roisNum: %d, output Format: %d ,output DataType: %d ", l2Para.roiParm.riosNum, l2Para.output.format, l2Para.output.dataType);
        table->roiMultiBatchNum = l2Para.input.batchSize;
        // multibatch within input batch
        table->multiBatchOffsetInput = (int64_t)l2Para.input.height * (int64_t)l2Para.input.width * (int64_t)l2Para.input.channel * (int64_t)l2Para.input.channel0 * sizeof(uint16_t);
        table->multiBatchOffsetRoi = ((int64_t)l2Para.roiParm.riosNum / l2Para.input.batchSize) * (int64_t)l2Para.input.channel0 * sizeof(uint16_t);
        if (KERNEL_TENSOR_FORMAT_C1HWNC0 == l2Para.output.format)
        {
            table->multiBatchOffsetOutput = ((int64_t)l2Para.roiParm.riosNum / l2Para.input.batchSize) * (int64_t)l2Para.output.channel0 * cceGetDataTypeSize(l2Para.output.dataType);
        }
        else
        {
            table->multiBatchOffsetOutput = ((int64_t)l2Para.roiParm.riosNum / l2Para.input.batchSize) * (int64_t)l2Para.roiParm.pooledH * (int64_t)l2Para.roiParm.pooledW * (int64_t)l2Para.input.channel * (int64_t)l2Para.input.channel0 * sizeof(uint16_t);
        }

        BuildRoiMultiBatchParams(flowTable, l2Para.roiParm.riosNum / l2Para.input.batchSize, 0);
        table->roiMultiBatchParams[1] = table->roiMultiBatchParams[0];
    }
    else
    {
        CCE_LOG(CC_LOG_INFO, "SaveCommonParams 1 batch multi-core ! roisNum: %d, output Format: %d ,output DataType: %d ", l2Para.roiParm.riosNum, l2Para.output.format, l2Para.output.dataType);
        int64_t roiOneBatchNum = CEIL((int64_t)l2Para.roiParm.riosNum, (int64_t)roipoolingPlatformPara.aicore_cnt); // two core in mimi platform, dual
        table->roiMultiBatchNum = CEIL((int64_t)l2Para.roiParm.riosNum, roiOneBatchNum);                            // now dual, mean this should always be 2.
        // multibatch within roi num
        table->multiBatchOffsetInput = 0;
        table->multiBatchOffsetRoi = roiOneBatchNum * (int64_t)l2Para.input.channel0 * sizeof(uint16_t);
        if (KERNEL_TENSOR_FORMAT_C1HWNC0 == l2Para.output.format)
        {
            table->multiBatchOffsetOutput = roiOneBatchNum * (int64_t)l2Para.output.channel0 * cceGetDataTypeSize(l2Para.output.dataType);
        }
        else
        {
            table->multiBatchOffsetOutput = roiOneBatchNum * (int64_t)l2Para.roiParm.pooledH * (int64_t)l2Para.roiParm.pooledW * (int64_t)l2Para.output.channel * (int64_t)l2Para.output.channel0 * cceGetDataTypeSize(l2Para.output.dataType);
        }
        int64_t roiOneBatchNumL = (int64_t)l2Para.roiParm.riosNum % roiOneBatchNum;
        if (0 == roiOneBatchNumL)
        {
            roiOneBatchNumL = roiOneBatchNum;
        }
        BuildRoiMultiBatchParams(flowTable, (uint32_t)roiOneBatchNum, 0);  //Calc the common batchNum
        BuildRoiMultiBatchParams(flowTable, (uint32_t)roiOneBatchNumL, 1); //calc the last batch num
    }
#endif
    //output addrs
    table->inputAddr = l2Para.inputAddr;
    table->roiAddr = l2Para.roiAddr;
    table->outputAddr = l2Para.outputAddr;

    fp16_t spatialScaleFp16;
    spatialScaleFp16 = l2Para.roiParm.spatialScale;

    fp16_t pooledHRecip;
    pooledHRecip = 1.0 / (float)l2Para.roiParm.pooledH;

    fp16_t pooledWRecip;
    pooledWRecip = 1.0 / (float)l2Para.roiParm.pooledW;

    float vectorscale = std::pow(2.0, 0);
    fp16_t data;
    data = vectorscale;

    table->scales[0] = spatialScaleFp16.val;
    table->scales[1] = pooledHRecip.val;
    table->scales[2] = pooledWRecip.val;
    table->scales[3] = data.val;

    float phRec = (float)1.0 / (float)l2Para.roiParm.pooledH;
    float pwRec = (float)1.0 / (float)l2Para.roiParm.pooledW;
    table->pooledHWRec[0] = *(uint32_t *)&phRec;
    table->pooledHWRec[1] = *(uint32_t *)&pwRec;
    fp16_t pointFive;
    pointFive = float(0.5);
    table->pointFive[0] = pointFive.val;

    for (uint32_t i = 0; i < MAX_POOL_SIZE; i++)
    {
        float index = float(i);
        fp16_t data;
        data = index;
        table->indexArr[i] = data.val;
    }

    table->roiNCHWConvXt = CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, 2 * BLOCKNUM, STRIDE_ZERO, STRIDE_ZERO, BLOCKNUM);
    table->roiCalcPosXt = CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, DEFAULT_VECTOR_XN_STRIDE, DEFAULT_VECTOR_XD_REPEAT, DEFAULT_VECTOR_XN_REPEAT, DATA_COUNT_OF_EACH_ROI);
    table->roiCONVXt = CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, DEFAULT_VECTOR_XN_STRIDE, DEFAULT_VECTOR_XD_REPEAT, DEFAULT_VECTOR_XN_REPEAT / sizeof(uint16_t), BLOCKNUM);
    table->roiS32SubXt = CalcXtForTwoSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, DEFAULT_VECTOR_XN_STRIDE, DEFAULT_VECTOR_XM_STRIDE, DEFAULT_VECTOR_XD_REPEAT, DEFAULT_VECTOR_XN_REPEAT, DEFAULT_VECTOR_XM_REPEAT, 2 * ROI_EACH_LOOP / INT32_NUM_EACH_REPEAT);
    table->roiBinHWDeqXt = CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, DEFAULT_VECTOR_XN_STRIDE, DEFAULT_VECTOR_XD_REPEAT / sizeof(uint16_t), DEFAULT_VECTOR_XN_REPEAT, 2 * 2 * DEFAULT_REPEAT_TIME);
    table->roiF16OneRepXt = CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, DEFAULT_VECTOR_XN_STRIDE, DEFAULT_VECTOR_XD_REPEAT, DEFAULT_VECTOR_XN_REPEAT, DEFAULT_REPEAT_TIME);
    table->roiCONVFXt = CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, DEFAULT_VECTOR_XN_STRIDE, DEFAULT_VECTOR_XD_REPEAT, DEFAULT_VECTOR_XN_REPEAT / sizeof(uint16_t), 2 * 2 * DEFAULT_REPEAT_TIME);
    table->s32Dup2RepXt = CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, STRIDE_ZERO, DEFAULT_VECTOR_XD_REPEAT, STRIDE_ZERO, 2);
    table->s32Dup4RepXt = CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, STRIDE_ZERO, DEFAULT_VECTOR_XD_REPEAT, STRIDE_ZERO, 4);
    table->binCmpXt = CalcXtForTwoSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, DEFAULT_VECTOR_XN_STRIDE, DEFAULT_VECTOR_XM_STRIDE, DEFAULT_VECTOR_XD_REPEAT, DEFAULT_VECTOR_XN_REPEAT, DEFAULT_VECTOR_XM_REPEAT, 2 * DEFAULT_REPEAT_TIME);

    table->hiBufClearXt = CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, STRIDE_ZERO, DEFAULT_VECTOR_XD_REPEAT, STRIDE_ZERO, 0);
    table->hiVmaxXt = CalcXtForTwoSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, DEFAULT_VECTOR_XN_STRIDE, DEFAULT_VECTOR_XM_STRIDE, DEFAULT_VECTOR_XD_REPEAT, DEFAULT_VECTOR_XN_REPEAT, DEFAULT_VECTOR_XM_REPEAT, 0);
    table->fp16DupXt = CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, STRIDE_ZERO, DEFAULT_VECTOR_XD_REPEAT, STRIDE_ZERO, DEFAULT_REPEAT_TIME);
    table->fp16MulXt = CalcXtForTwoSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, DEFAULT_VECTOR_XN_STRIDE, DEFAULT_VECTOR_XM_STRIDE, DEFAULT_VECTOR_XD_REPEAT, DEFAULT_VECTOR_XN_REPEAT, DEFAULT_VECTOR_XM_REPEAT, DEFAULT_REPEAT_TIME);
}
void RoipoolingPerfForwardKernel::CalcRegScenario128C0(void *flowTable)
{
    kRoipoolingPerfFlowTable_t *table = (kRoipoolingPerfFlowTable_t *)flowTable;

    // ub addrs
    table->ubFeaturMapAddr = FEATUR_MAP_ADDR_UB;
    table->ubResultAddr[0] = POOLING_RESULT_ADDR_UB;
    table->ubResultAddr[1] = POOLING_RESULT_ADDR_UB + PINGPONG_BUFF_OFFSET;
    table->ubHiLoopBufAddr[0] = HI_LOOP_BUF;
    table->ubHiLoopBufAddr[1] = HI_LOOP_BUF + PINGPONG_BUFF_OFFSET;
    table->ubPosAddr = POS_ADDR_UB;
    table->ubFlowtableAddr = FLOWTABLE_ADDR;

    // input
    table->ciLoops = CEIL(l2Para.input.channel, BLOCKNUM);

    // load multipule times for each c0 block;
    table->loadTimes = BLOCKNUM;
    table->loadTimesL = l2Para.input.channel % BLOCKNUM;
    if (0 == table->loadTimesL)
    {
        table->loadTimesL = table->loadTimes;
    }

    uint64_t burstNum, burstLen, srcStride, dstStride = 0;
    burstNum = (uint64_t)l2Para.input.height * l2Para.input.width;
    burstLen = (uint64_t)DEFULT_MOV_BURSTS;
    srcStride = (uint64_t)STRIDE_ZERO;
    dstStride = (uint64_t)BLOCKNUM - 1;
    table->inputXm = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
    table->inputOffset = (uint64_t)l2Para.input.height * l2Para.input.width * l2Para.input.channel0;

    table->clearResultXt = CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, STRIDE_ZERO, DEFAULT_VECTOR_XD_REPEAT, STRIDE_ZERO, (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW);
    table->roiVmaxXt = CalcXtForTwoSrcVectorOP((uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW, DEFAULT_VECTOR_XN_STRIDE, (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW, STRIDE_ZERO, DEFAULT_VECTOR_XN_REPEAT, STRIDE_ZERO, 0);

    //offsets
    table->inputWOffSet = (int64_t)l2Para.input.width * ROI_EACH_LOOP;
    table->outputWOffset = (int64_t)l2Para.roiParm.pooledW * (int64_t)l2Para.input.channel0;
    table->pooledHOffset = (int64_t)l2Para.roiParm.pooledH * ROI_EACH_LOOP;
    table->pooledWOffset = (int64_t)l2Para.roiParm.pooledW * ROI_EACH_LOOP;

    table->outCiOffset = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * CHANNEL_EACH_LOOP;
    table->roiOutOffset = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * l2Para.input.channel * l2Para.input.channel0;
    table->roiGroupOffset = table->roiOutOffset * CHANNEL_EACH_LOOP;

    uint32_t outputC1Num = l2Para.input.channel > BLOCKNUM ? BLOCKNUM : l2Para.input.channel;
    burstNum = (uint64_t)DEFULT_MOV_BURSTS;
    burstLen = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * outputC1Num;
    srcStride = (uint64_t)STRIDE_ZERO;
    dstStride = (uint64_t)STRIDE_ZERO;
    table->outputXm = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
    burstLen = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * REMAINDER(l2Para.input.channel, 8);
    table->outputXmL = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
    if (KERNEL_TENSOR_FORMAT_C1HWNC0 == l2Para.output.format)
    {
#ifndef __CCE_KT_TEST__
        CCE_LOG(CC_LOG_INFO, "roipooling output C1HWNC0 for Fc!");
        table->outCiOffset = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * CHANNEL_EACH_LOOP * l2Para.roiParm.riosNum;
        table->roiOutOffset = C0SIZE;//table->outCiOffset;
        table->roiGroupOffset = C0SIZE * CHANNEL_EACH_LOOP;//table->outCiOffset * CHANNEL_EACH_LOOP;

        burstNum = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * BLOCKNUM;
        burstLen = (uint64_t)DEFULT_MOV_BURSTS_LENGTH;
        srcStride = (uint64_t)STRIDE_ZERO;
        dstStride = l2Para.roiParm.riosNum - 1;
        table->outputXm = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
        burstNum = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * REMAINDER(l2Para.input.channel, 8);
        table->outputXmL = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
#endif
    }
}

void RoipoolingPerfForwardKernel::CalcRegScenarioOneC0(void *flowTable)
{
    kRoipoolingPerfFlowTable_t *table = (kRoipoolingPerfFlowTable_t *)flowTable;
    /**************************************/
    /*          UB POS                    */
    /**************************************/
    /*          FeatureMap                */
    /**************************************/
    /*          Result[0]                 */
    /**************************************/
    /*         HiLoop Buffer              */
    /**************************************/
    /*         Result[1]                  */
    /**************************************/
    /*           FlowTable  8K            */
    /**************************************/
    /*           Compiler  8K             */
    /**************************************/
    // ub addrs
    table->ubPosAddr = SCEN_ONEC0_POS_ADDR;
    table->ubFeaturMapAddr = SCEN_ONEC0_FEATUR_MAP_ADDR;
    table->ubResultAddr[0] = SCEN_ONEC0_RESULT_PING_ADDR;
    table->ubResultAddr[1] = SCEN_ONEC0_RESULT_PONG_ADDR;
    //table->ubResultAddr[1] = SCEN_ONEC0_RESULT_PING_ADDR;

    // no hi pingpong buf this scenario
    table->ubHiLoopBufAddr[0] = SCEN_ONEC0_HI_LOOP_BUF;
    table->ubHiLoopBufAddr[1] = SCEN_ONEC0_HI_LOOP_BUF;
    table->ubFlowtableAddr = SCEN_ONEC0_FLOWTABLE_ADDR;

    // input
    table->ciLoops = l2Para.input.channel;
    // load one c0 block each time.
    uint64_t burstNum, burstLen, srcStride, dstStride = 0;
    burstNum = 1;
    burstLen = (uint64_t)l2Para.input.height * l2Para.input.width;
    srcStride = (uint64_t)STRIDE_ZERO;
    dstStride = (uint64_t)STRIDE_ZERO;
    table->inputXm = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
    table->inputOffset = (uint64_t)l2Para.input.height * l2Para.input.width * l2Para.input.channel0;

    uint64_t outputHW = (int64_t)l2Para.roiParm.pooledH * (int64_t)l2Para.roiParm.pooledW;
    uint64_t outHWRepeat = CEIL(outputHW, BLOCKNUM);
    table->clearResultXt = CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, STRIDE_ZERO, DEFAULT_VECTOR_XD_REPEAT, STRIDE_ZERO, outHWRepeat);

    table->repeatPH = CEIL(l2Para.roiParm.pooledH, BLOCKNUM);

    //XN stride need update by currRoiW, and XN repeat stride is pooledH * currRoiW
    table->roiVmaxXt = CalcXtForTwoSrcVectorOP(l2Para.roiParm.pooledW, STRIDE_ZERO, l2Para.roiParm.pooledW, 0, 1, 0, 0);

    //offsets
    table->inputWOffSet = (int64_t)l2Para.input.width * (int64_t)l2Para.input.channel0;
    table->outputWOffset = (int64_t)l2Para.roiParm.pooledW * (int64_t)l2Para.input.channel0;
    table->pooledHOffset = (int64_t)l2Para.roiParm.pooledH * ROI_EACH_LOOP;
    table->pooledWOffset = (int64_t)l2Para.roiParm.pooledW * ROI_EACH_LOOP;

    table->outCiOffset = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * l2Para.input.channel0;
    table->roiOutOffset = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * l2Para.input.channel * l2Para.input.channel0;
    table->roiGroupOffset = table->roiOutOffset * ROI_EACH_LOOP;
    burstNum = (uint64_t)DEFULT_MOV_BURSTS;
    burstLen = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW;
    srcStride = (uint64_t)STRIDE_ZERO;
    dstStride = (uint64_t)STRIDE_ZERO;

    if (KERNEL_TENSOR_FORMAT_C1HWNC0 == l2Para.output.format)
    {
#ifndef __CCE_KT_TEST__
        CCE_LOG(CC_LOG_INFO, "roipooling output C1HWNC0 for Fc!");
        table->outCiOffset = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * l2Para.input.channel0 * l2Para.roiParm.riosNum;
        table->roiOutOffset = (uint64_t)16;
        table->roiGroupOffset = table->roiOutOffset * ROI_EACH_LOOP;

        burstNum = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW;
        burstLen = (uint64_t)1;
        srcStride = (uint64_t)STRIDE_ZERO;
        dstStride = (uint64_t)l2Para.roiParm.riosNum - 1;
#endif
    }

    table->outputXm = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
    table->outputXmL = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
}

void RoipoolingPerfForwardKernel::CalcRegScenarioFourC0(void *flowTable)
{
    kRoipoolingPerfFlowTable_t *table = (kRoipoolingPerfFlowTable_t *)flowTable;

    // ub addrs
    table->ubPosAddr = SCEN_ONEC0_POS_ADDR;
    table->ubFeaturMapAddr = SCEN_ONEC0_FEATUR_MAP_ADDR;
    table->ubResultAddr[0] = SCEN_ONEC0_RESULT_PING_ADDR;
    table->ubResultAddr[1] = SCEN_ONEC0_RESULT_PONG_ADDR;

    // no hi pingpong buf this scenario
    table->ubHiLoopBufAddr[0] = SCEN_ONEC0_HI_LOOP_BUF;
    table->ubHiLoopBufAddr[1] = SCEN_ONEC0_HI_LOOP_BUF;
    table->ubFlowtableAddr = SCEN_ONEC0_FLOWTABLE_ADDR;

    // input
    table->ciLoops = CEIL(l2Para.input.channel, 4);
    // load one c0 block each time.
    uint64_t burstNum, burstLen, srcStride, dstStride = 0;
    burstNum = 1;
    burstLen = (uint64_t)l2Para.input.height * l2Para.input.width * 4;
    srcStride = (uint64_t)STRIDE_ZERO;
    dstStride = (uint64_t)STRIDE_ZERO;
    table->inputXm = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
    burstLen = (uint64_t)l2Para.input.height * l2Para.input.width * REMAINDER(l2Para.input.channel, 4);
    table->inputXmL = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
    table->inputOffset = (uint64_t)l2Para.input.height * l2Para.input.width * l2Para.input.channel0 * 4;
    table->inputOffsetC0 = (uint64_t)l2Para.input.height * l2Para.input.width * l2Para.input.channel0;

    uint64_t outputHW = (int64_t)l2Para.roiParm.pooledH * (int64_t)l2Para.roiParm.pooledW * 4;
    uint64_t outHWRepeat = CEIL(outputHW, BLOCKNUM);
    table->clearResultXt = CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, STRIDE_ZERO, DEFAULT_VECTOR_XD_REPEAT, STRIDE_ZERO, outHWRepeat);

    table->repeatPH = CEIL((int64_t)l2Para.roiParm.pooledH * 4, BLOCKNUM);

    //XN stride need update by currRoiW, and XN repeat stride is pooledH * currRoiW
    table->roiVmaxXt = CalcXtForTwoSrcVectorOP(l2Para.roiParm.pooledW, STRIDE_ZERO, l2Para.roiParm.pooledW, 0, 1, 0, 0);

    //offsets
    table->inputWOffSet = (int64_t)l2Para.input.width * (int64_t)l2Para.input.channel0;
    table->outputWOffset = (int64_t)l2Para.roiParm.pooledW * (int64_t)l2Para.input.channel0;
    table->pooledHOffset = (int64_t)l2Para.roiParm.pooledH * ROI_EACH_LOOP;
    table->pooledWOffset = (int64_t)l2Para.roiParm.pooledW * ROI_EACH_LOOP;

    if (KERNEL_DATA_TYPE_INT8 == l2Para.output.dataType || KERNEL_DATA_TYPE_UINT8 == l2Para.output.dataType)
    {
        CCE_LOG(CC_LOG_INFO, "roipooling output C1HWNC0&INT8 for Fc!");
        table->needQuant = QUANT_INT8;

        table->int8Scale = (uint64_t)l2Para.roiParm.quantizeParaInfo.scaleQ.scale;
        table->int8Offset = (uint64_t)l2Para.roiParm.quantizeParaInfo.scaleQ.offsetq;

        table->convF162S8Xt = CalcXtForOneSrcVectorOP(1, 1, 4, 8, CEIL((uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * 4, 8));
        table->quantScaleXt = CalcXtForOneSrcVectorOP(2, 1, 16, 8, CEIL((uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW, 8));
        table->quantOffsetXt = CalcXtForOneSrcVectorOP(1, 1, 8, 8, CEIL((uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * 4, 8));

        table->outCiOffset = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * l2Para.input.channel0 * 4;
        table->roiOutOffset = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * l2Para.input.channel * l2Para.input.channel0;
        table->roiGroupOffset = table->roiOutOffset * ROI_EACH_LOOP;

        burstNum = (uint64_t)DEFULT_MOV_BURSTS;
        burstLen = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * 2;
        srcStride = (uint64_t)STRIDE_ZERO;
        dstStride = (uint64_t)STRIDE_ZERO;
        table->outputXm = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
        table->outputXmL = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
        if (KERNEL_TENSOR_FORMAT_C1HWNC0 == l2Para.output.format)
        {
#ifndef __CCE_KT_TEST__
            CCE_LOG(CC_LOG_INFO, "roipooling output C1HWNC0 for Fc!");

            table->outCiOffset = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * l2Para.input.channel0 * l2Para.roiParm.riosNum * 4;
            table->roiOutOffset = (uint64_t)32;
            table->roiGroupOffset = table->roiOutOffset * ROI_EACH_LOOP;

            burstNum = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * 2;
            burstLen = (uint64_t)DEFULT_MOV_BURSTS_LENGTH;
            srcStride = (uint64_t)STRIDE_ZERO;
            dstStride = (uint64_t)l2Para.roiParm.riosNum - 1;
            table->outputXm = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
            table->outputXmL = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
#endif
        }
    }
    else
    {
        table->needQuant = QUANT_NONE;

        table->outCiOffset = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * l2Para.input.channel0 * 4;
        table->roiOutOffset = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * l2Para.input.channel * l2Para.input.channel0;
        table->roiGroupOffset = table->roiOutOffset * ROI_EACH_LOOP;

        burstNum = (uint64_t)DEFULT_MOV_BURSTS;
        burstLen = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * 4;
        srcStride = (uint64_t)STRIDE_ZERO;
        dstStride = (uint64_t)STRIDE_ZERO;
        table->outputXm = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
        burstLen = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * REMAINDER(l2Para.input.channel, 4);
        table->outputXmL = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
        if (KERNEL_TENSOR_FORMAT_C1HWNC0 == l2Para.output.format)
        {
#ifndef __CCE_KT_TEST__
            CCE_LOG(CC_LOG_INFO, "roipooling output C1HWNC0 for Fc!");
            table->outCiOffset = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * l2Para.input.channel0 * l2Para.roiParm.riosNum * 4;
            table->roiOutOffset = (uint64_t)16;
            table->roiGroupOffset = table->roiOutOffset * ROI_EACH_LOOP;

            burstNum = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * 4;
            burstLen = (uint64_t)1;
            srcStride = (uint64_t)STRIDE_ZERO;
            dstStride = (uint64_t)l2Para.roiParm.riosNum - 1;
            table->outputXm = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
            burstNum = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * REMAINDER(l2Para.input.channel, 4);
            table->outputXmL = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
#endif
        }
    }
}

void RoipoolingPerfForwardKernel::CalcRegScenarioL1(void *flowTable)
{
    kRoipoolingPerfFlowTable_t *table = (kRoipoolingPerfFlowTable_t *)flowTable;

    // ub addrs
    table->ubPosAddr = SCEN_ONEC0_POS_ADDR;
    table->ubFeaturMapAddr = SCEN_ONEC0_FEATUR_MAP_ADDR;
    table->ubResultAddr[0] = SCEN_ONEC0_RESULT_PING_ADDR;
    table->ubResultAddr[1] = SCEN_ONEC0_RESULT_PONG_ADDR;
    //table->ubResultAddr[1] = SCEN_ONEC0_RESULT_PING_ADDR;

    // no hi pingpong buf this scenario
    table->ubHiLoopBufAddr[0] = SCEN_ONEC0_HI_LOOP_BUF;
    table->ubHiLoopBufAddr[1] = SCEN_ONEC0_HI_LOOP_BUF;
    table->ubFlowtableAddr = SCEN_ONEC0_FLOWTABLE_ADDR;

    // input
    table->ciLoops = l2Para.input.channel;
    table->c1NumInL1 = (uint64_t)roipoolingPlatformPara.l1Size / ((uint64_t)l2Para.input.height * (uint64_t)l2Para.input.width * (uint64_t)l2Para.input.channel0 * sizeof(uint16_t));

    // load one c0 block each time.
    uint64_t burstNum, burstLen, srcStride, dstStride = 0;
    burstNum = 1;
    burstLen = table->c1NumInL1 * (uint64_t)l2Para.input.height * l2Para.input.width;
    srcStride = (uint64_t)STRIDE_ZERO;
    dstStride = (uint64_t)STRIDE_ZERO;
    table->inputXm = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
    table->inputOffset = (uint64_t)l2Para.input.height * l2Para.input.width * l2Para.input.channel0;

    uint64_t outputHW = (int64_t)l2Para.roiParm.pooledH * (int64_t)l2Para.roiParm.pooledW;
    uint64_t outHWRepeat = CEIL(outputHW, BLOCKNUM);
    table->clearResultXt = CalcXtForOneSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, STRIDE_ZERO, DEFAULT_VECTOR_XD_REPEAT, STRIDE_ZERO, outHWRepeat);

    table->repeatPH = CEIL(l2Para.roiParm.pooledH, BLOCKNUM);

    table->hiVmaxXt = CalcXtForTwoSrcVectorOP(DEFAULT_VECTOR_XD_STRIDE, DEFAULT_VECTOR_XN_STRIDE, DEFAULT_VECTOR_XM_STRIDE, 0, 0, 0, 0);
    //XN stride need update by currRoiW, and XN repeat stride is pooledH * currRoiW
    table->roiVmaxXt = CalcXtForTwoSrcVectorOP(l2Para.roiParm.pooledW, STRIDE_ZERO, l2Para.roiParm.pooledW, 0, 1, 0, 0);
    //offsets
    table->inputWOffSet = (int64_t)l2Para.input.width * (int64_t)l2Para.input.channel0;
    table->outputWOffset = (int64_t)l2Para.roiParm.pooledW * (int64_t)l2Para.input.channel0;
    table->pooledHOffset = (int64_t)l2Para.roiParm.pooledH * ROI_EACH_LOOP;
    table->pooledWOffset = (int64_t)l2Para.roiParm.pooledW * ROI_EACH_LOOP;

    table->outCiOffset = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * l2Para.input.channel0;
    table->roiOutOffset = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * l2Para.input.channel * l2Para.input.channel0;
    table->roiGroupOffset = table->roiOutOffset * ROI_EACH_LOOP;
    burstNum = (uint64_t)DEFULT_MOV_BURSTS;
    burstLen = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW;
    srcStride = (uint64_t)STRIDE_ZERO;
    dstStride = (uint64_t)STRIDE_ZERO;

    if (KERNEL_TENSOR_FORMAT_C1HWNC0 == l2Para.output.format)
    {
#ifndef __CCE_KT_TEST__
        CCE_LOG(CC_LOG_INFO, "roipooling output C1HWNC0 for Fc!");
        table->outCiOffset = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW * l2Para.input.channel0 * l2Para.roiParm.riosNum;
        table->roiOutOffset = (uint64_t)16;
        table->roiGroupOffset = table->roiOutOffset * ROI_EACH_LOOP;

        burstNum = (uint64_t)l2Para.roiParm.pooledH * l2Para.roiParm.pooledW;
        burstLen = (uint64_t)1;
        srcStride = (uint64_t)STRIDE_ZERO;
        dstStride = (uint64_t)l2Para.roiParm.riosNum - 1;
#endif
    }

    table->outputXm = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
    table->outputXmL = CalcXmForMov(burstNum, burstLen, srcStride, dstStride);
}

uint32_t RoipoolingPerfForwardKernel::BuildFlowTablePara(void *flowTable)
{
    kRoipoolingPerfFlowTable_t *table = (kRoipoolingPerfFlowTable_t *)flowTable;

    memset_s((void *)table, sizeof(kRoipoolingPerfFlowTable_t), 0x00, sizeof(kRoipoolingPerfFlowTable_t));

    SaveCommonParams(flowTable);
    uint32_t scenario = PERF_SCENARIO_NA;
    //Mini platform
#ifdef DAVINCI_MINI
    uint64_t c0BlockSize = (int64_t)l2Para.input.height * (int64_t)l2Para.input.width * (int64_t)l2Para.input.channel0 * sizeof(uint16_t);
    if ((BLOCKNUM * c0BlockSize < FM_MAX_LOAD_SIZE) && (l2Para.output.height * l2Para.output.width <= SENC0_MIN_POOLEDHW))
    {
        // 128 parallel in channel direction.
        scenario = PERF_SCENARIO_128_C0;
        CalcRegScenario128C0(flowTable);
    }
    else if ((l2Para.input.height * l2Para.input.width <= 920) && (l2Para.output.height * l2Para.output.width <= 36))
    {
        // process four c0 each time. resnet18 faster-rcnn perf branch
        scenario = PERF_SCENARIO_FOUR_C0;
        CalcRegScenarioFourC0(flowTable);
    }
    else if ((c0BlockSize <= SCEN_ONEC0_FM_MAX_LOAD_SIZE) && (l2Para.output.height * l2Para.output.width <= MAX_POOLEDHW))
    {
        // process one c0 each time.
        scenario = PERF_SCENARIO_ONE_C0;
        CalcRegScenarioOneC0(flowTable);
    }
    else if (l2Para.output.height * l2Para.output.width <= PERF_MAX_FM_L1)
    {
        scenario = PERF_SCENARIO_L1;
        CalcRegScenarioL1(flowTable);
    }
#ifndef __CCE_KT_TEST__
    else
    {
        CCE_LOG(CC_LOG_ERROR, "GenFlowTable fail, input Size is too big! inputH: %d, inputW: %d", l2Para.input.height, l2Para.input.width);
        return CCE_FAIL;
    }
#endif
#elif defined(DAVINCI_LITE)
    // process one c0 each time.
    scenario = PERF_SCENARIO_ONE_C0;
    CalcRegScenarioOneC0(flowTable);
#elif defined(DAVINCI_TINY)
    scenario = PERF_SCENARIO_ONE_C0;
    CalcRegScenarioOneC0(flowTable);
#endif

    table->scenario = scenario;
    //L2 addr
    table->enableL2 = 0;
    taskL2Info.enableL2 = 0;
    char *fusenv = getenv(g_l2fusion);
    if ((NULL != fusenv) && (0 == strcmp("1", fusenv)))
    {
        table->enableL2 = 1;
        taskL2Info.enableL2 = 1;
    }
    memcpy_s(&table->inputL2AddrInfo, sizeof(VecL2AddrInfo), &taskL2Info.inputL2AddrInfo, sizeof(VecL2AddrInfo));
    memcpy_s(&table->roisL2AddrInfo, sizeof(VecL2AddrInfo), &taskL2Info.roisL2AddrInfo, sizeof(VecL2AddrInfo));
    memcpy_s(&table->outputL2AddrInfo, sizeof(VecL2AddrInfo), &taskL2Info.outputL2AddrInfo, sizeof(VecL2AddrInfo));

    return CCE_SUCCESS;
}

uint32_t RoipoolingPerfForwardKernel::GetBlockDim(AttrList &attrList, uint32_t &x)
{
    uint32_t attrLen = 0;
    const void *pAttrValue;

    //get input tensor
    kTensor_t *inputTensor = NULL;
    attrList.Get(ROIPOOLING_KERNEL_TENSOR_INPUT, attrLen, pAttrValue);
    inputTensor = (kTensor_t *)pAttrValue;

    attrList.Get(ROIPOOLING_KERNEL_PARAM, attrLen, pAttrValue);
    kRoipoolingParam_t *prarm = (kRoipoolingParam_t *)pAttrValue;

    if (1 == inputTensor->batchSize && 2 == roipoolingPlatformPara.aicore_cnt)
    {
        x = (prarm->riosNum == 1) ? 1 : 2; // mini platform has two core now, split roi
    }
#ifndef __CCE_KT_TEST__
    else
    {
        x = inputTensor->batchSize;
    }
#endif
    return CCE_SUCCESS;
}

uint32_t RoipoolingPerfForwardKernel::GetEntryPoint(uint64_t &entryAddr)
{
    (void)entryAddr;
    return CCE_SUCCESS;
}

uint32_t RoipoolingPerfForwardKernel::GetPreloadData(AttrList &attrList, AttrList &dataAddrList, rtL2Ctrl_t &l2ctrl)
{
    (void)attrList;
    (void)dataAddrList; //Clear static check:unused paramters.
    l2ctrl = taskL2Info.remapTbl;
    l2ctrl.preloadOffset = 0;
    l2ctrl.size = taskL2Info.maxPageNum;
    l2ctrl.preloadSize = 0;
    l2ctrl.preloadSrc = 0;
    if (!taskL2Info.enableL2)
    {
        memset_s(&l2ctrl, sizeof(rtL2Ctrl_t), 0, sizeof(rtL2Ctrl_t));
    }
    printL2RemaptblInfo(l2ctrl);
    return CCE_SUCCESS;
}
void RoipoolingPerfForwardKernel::SetPlatformParam()
{
    roipoolingPlatformPara.aicore_cnt = ccGetAiCoreCnt();
    roipoolingPlatformPara.l2Size = ccGetL2Size();
    roipoolingPlatformPara.ubSize = ccGetUBSize();
    roipoolingPlatformPara.l1Size = ccGetL1Size();
    roipoolingPlatformPara.compilerUBSize = ccGetCompilerSize();
    roipoolingPlatformPara.flowtableUBSize = ccGetFlowtableSize();
    roipoolingPlatformPara.platformConfig = (uint64_t)ccGetPlatformConfig();
}

//add for ST

#ifdef ENABLE_MIX_COMPILE
extern void run_executor_roipooling_perf(uint32_t coreDim, void *l2ctrl, void *stream, void *tableAddr, uint32_t tableSize);

void RoipoolingPerfForwardKernel::run_executor(uint32_t coreDim, void *l2ctrl, void *stream, void *tableAddr, uint32_t tableSize)
{
    run_executor_roipooling_perf(coreDim, l2ctrl, stream, tableAddr, tableSize);
}
#endif

void RoipoolingPerfForwardKernel::getOpInfo(kOpInfo &opInfo)
{
    opInfo.opId = CCE_DNN_OP_ROIPOOLING;
    opInfo.mode = true;
    opInfo.opIdsLen = 0;
}

}; // namespace kernel
}; // namespace cce

```
