template<int halfSpan, int SDC_FRAME_SIZE, int tempSize >
__global__ void GPU_OPT_EmbeddedAccumulate(float* __restrict__ inArray,
    float* __restrict__ outArray,
    float* __restrict__ pixelVal,
    float* __restrict__ k1D)
{
    int rowNum = blockDim.x + 2 * halfSpan;
    int colNum = blockDim.y + 2 * halfSpan;
    float temp1;
    float temp2;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int rowTemp = threadIdx.y + halfSpan;
    int ColTemp = threadIdx.x + halfSpan;
    __shared__ float temp[1024];
    __shared__ float tempArr[4352];
    __shared__ float tempk1d[17];

    if (col < SDC_FRAME_SIZE && row < SDC_FRAME_SIZE) {
        if (threadIdx.y == 0 || (threadIdx.y == 1 && threadIdx.x == 0)) {
            tempk1d[threadIdx.x] = k1D[threadIdx.x];
            if (threadIdx.y == 1) {
                tempk1d[16] = k1D[16];
            }
        }
        __syncthreads();

        if (row < halfSpan || row >= (SDC_FRAME_SIZE - halfSpan)) {
            outArray[row * SDC_FRAME_SIZE + col] = 0;
            temp[rowTemp * rowNum + ColTemp] = inArray[row * SDC_FRAME_SIZE + col];
            if (threadIdx.y < halfSpan && threadIdx.x < halfSpan ) {
                temp[(rowTemp + blockDim.y) * rowNum + ColTemp - halfSpan] = inArray[(row + blockDim.y) * SDC_FRAME_SIZE + col - halfSpan];
                temp[(rowTemp + blockDim.y) * rowNum + ColTemp + blockDim.x] = inArray[(row + blockDim.y) * SDC_FRAME_SIZE + col + blockDim.x];
            }
        }
        else {

            temp[rowTemp * rowNum + ColTemp] = inArray[row * SDC_FRAME_SIZE + col];
            if (threadIdx.y < halfSpan && threadIdx.x < halfSpan) {
                if (row >= halfSpan && row < halfSpan * 2 ) {
                
                }
                else if (row >= (SDC_FRAME_SIZE - (halfSpan * 2))) {

                    temp[(rowTemp - halfSpan) * rowNum + ColTemp - halfSpan] = inArray[(row - halfSpan) * SDC_FRAME_SIZE + col - halfSpan];
                    temp[(rowTemp - halfSpan) * rowNum + ColTemp + blockDim.x] = inArray[(row - halfSpan) * SDC_FRAME_SIZE + col + blockDim.x];

                }
                else {
                   
                    temp[(rowTemp - halfSpan) * rowNum + ColTemp - halfSpan] = inArray[(row - halfSpan) * SDC_FRAME_SIZE + col - halfSpan];
                    temp[(rowTemp - halfSpan) * rowNum + ColTemp + blockDim.x] = inArray[(row - halfSpan) * SDC_FRAME_SIZE + col + blockDim.x];

                    temp[(rowTemp + blockDim.y) * rowNum + ColTemp - halfSpan] = inArray[(row + blockDim.y) * SDC_FRAME_SIZE + col - halfSpan];
                    temp[(rowTemp + blockDim.y) * rowNum + ColTemp + blockDim.x] = inArray[(row + blockDim.y) * SDC_FRAME_SIZE + col + blockDim.x];
                }
            }
            if (row >= halfSpan && row < halfSpan * 2) {
                temp[(rowTemp + halfSpan) * rowNum + ColTemp] = inArray[(row + halfSpan) * SDC_FRAME_SIZE + col];
            }
            else if (row >= (SDC_FRAME_SIZE - (halfSpan * 2))) {
                temp[(rowTemp - halfSpan) * rowNum + ColTemp] = inArray[(row - halfSpan) * SDC_FRAME_SIZE + col];
 
            }
            else if (threadIdx.y < halfSpan) {

               temp[(rowTemp - halfSpan)* rowNum + ColTemp] = inArray[(row - halfSpan) * SDC_FRAME_SIZE + col];
               temp[(rowTemp + (halfSpan * 2)) * rowNum + ColTemp] = inArray[(row + (halfSpan * 2)) * SDC_FRAME_SIZE + col];
            }

                if (col >= halfSpan && col < halfSpan * 2) {
                   
                    temp[rowTemp * rowNum + ColTemp + halfSpan] = inArray[row * SDC_FRAME_SIZE + col + halfSpan]; 
                }
                else if (col >= (SDC_FRAME_SIZE - (halfSpan * 2))) {
                    temp[rowTemp * rowNum + ColTemp - halfSpan] = inArray[row * SDC_FRAME_SIZE + col - halfSpan];


                }
  
                else if (threadIdx.x < halfSpan) {
                  
                    temp[rowTemp * rowNum + ColTemp - halfSpan] = inArray[row * SDC_FRAME_SIZE + col - halfSpan];
                    temp[rowTemp * rowNum + ColTemp + (halfSpan * 2)] = inArray[row * SDC_FRAME_SIZE + col + (halfSpan * 2)];
                }
           
            __syncthreads();
           
            if (col < halfSpan || col >= (SDC_FRAME_SIZE - halfSpan)) {
                
            }
            else {
                for (int i = 0; i < 17; i++) {
                    tempArr[(threadIdx.y * 16 * 17) + (threadIdx.x* 17) + i] = tempk1d[halfSpan] * temp[rowTemp * rowNum + ColTemp - 8 + i];
 
                    for (int k = -halfSpan; k < 0; k++) {
                        tempArr[(threadIdx.y * 16 * 17) + (threadIdx.x * 17) + i] += tempk1d[k + halfSpan] * ( temp[(rowTemp+k) * rowNum + ColTemp - 8 + i] + temp[(rowTemp - k) * rowNum + ColTemp - 8 + i]);
                       
                    }
                }
            }

            __syncthreads();

            if (col < halfSpan || col >= (SDC_FRAME_SIZE - halfSpan) ) {
                outArray[row * SDC_FRAME_SIZE + col] = 0;
            }
            else {
                temp2 = tempk1d[halfSpan] * tempArr[threadIdx.y * 16 * 17 + (threadIdx.x * 17) + 8];

                for (int k = -halfSpan; k < 0; k++) {
                    temp2 += tempk1d[k + halfSpan] * (tempArr[threadIdx.y * 16 * 17+ (threadIdx.x * 17) + 8 - k] + tempArr[threadIdx.y * 16 * 17 + (threadIdx.x * 17) + 8 + k]);
                }

                outArray[row * SDC_FRAME_SIZE + col] = temp2;
            }
        }
    }
}