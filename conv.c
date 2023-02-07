template<int halfSpan>
void SpaceDomainConvolve_v2::EmbeddedAccumulate(float *__restrict__ * __restrict__ inArray,
                                                float *__restrict__ * __restrict__ outArray,
                                                float * __restrict__ pixelVal,
                                                SpaceDomainKernel *  __restrict__ gKernel)
{
  int rowSize = SDC_FRAME_SIZE * sizeof(float);
  memset((outArray[0]), '\0', halfSpan * SDC_FRAME_SIZE * sizeof(float));

  for(int row=halfSpan; row<SDC_FRAME_SIZE-halfSpan; row++)
  {
    float * rowPtr = &(inArray[row][0]);
    float * outPtr = &(outArray[row][0]);
#if (__GNUC__ > 5)
    // Both arrays are aligned at AVX2_ALIGNMENT_SIZE boundary. Make this claim to help AVX2 vectorization faster.
    rowPtr = (float *) CLAIM_ALIGNMENT(rowPtr);
    outPtr = (float *) CLAIM_ALIGNMENT(outPtr);
#endif

    // Do the column-wise convolution for all columns in this row
    // using the kernel as a vertical 1D array
    for(int col=0; col<SDC_FRAME_SIZE; col++, outPtr++, rowPtr++)
    {
      *outPtr = gKernel->k1D[halfSpan] * *(rowPtr);
      for(int k = -halfSpan, incr = k * SDC_FRAME_SIZE; k<0; k++, incr += SDC_FRAME_SIZE)
        *outPtr += gKernel->k1D[k+halfSpan] * (*(rowPtr+incr) + *(rowPtr-incr));
    }

    float * colPtr = &(outArray[row][halfSpan]);
    float * pixPtr = &(pixelVal[halfSpan]);

    // Do the row-wise convolution immediately for this row as the contiguous pixels
    // of full row can be expected to be in cache. Here the kernel is used
    // as a horizontal 1D array.
    for(int col = halfSpan; col<SDC_FRAME_SIZE-halfSpan; col++, pixPtr++, colPtr++)
    {
      *pixPtr = gKernel->k1D[halfSpan] * *(colPtr);
      for(int k = -halfSpan; k < 0; k++)
        *pixPtr += gKernel->k1D[k+halfSpan] * (*(colPtr+k) + *(colPtr-k));
    }
    memcpy(outArray[row],&(pixelVal[0]),rowSize);
  }
  memset(outArray[SDC_FRAME_SIZE-halfSpan], '\0', halfSpan * SDC_FRAME_SIZE * sizeof(float));
}

