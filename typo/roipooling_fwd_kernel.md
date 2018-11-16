```
    uint64_t outputAddr = *((uint64_t *)attrValue);

    l2Para.input.batchSize = inputTensor->batchSize;
    l2Para.input.height = inputTensor->height;
    l2Para.input.width = inputTensor->width;
    l2Para.input.channel = inputTensor->channel;
    l2Para.input.channel0 = inputTensor->channel0;
    l2Para.input.padC0 = inputTensor->padC0;

    l2Para.output.batchSize = outputTensor->batchSize;
    l2Para.output.height = outputTensor->height;
    l2Para.output.width = outputTensor->width;
    l2Para.output.channel = outputTensor->channel;
    l2Para.output.channel0 = outputTensor->channel0;
    l2Para.output.padC0 = outputTensor->padC0;

    l2Para.roiParm.poolingMode = prarm->poolingMode;
    l2Para.roiParm.pooledH = prarm->pooledH;
    l2Para.roiParm.pooledW = prarm->pooledW;
    l2Para.roiParm.spatialScale = prarm->spatialScale;
    l2Para.roiParm.riosNum = prarm->riosNum;

    l2Para.inputAddr = inputAddr;
    l2Para.outputAddr = outputAddr;
    l2Para.roiAddr = roisAddr;
    return CCE_SUCCESS;
```
