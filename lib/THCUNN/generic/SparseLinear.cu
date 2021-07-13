#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SparseLinear.cu"
#else

static bool checkInput(THCTensor* t)
{
  return t->nDimension == 2 && t->size[1] == 3;
}

static bool checkSize2D(THCTensor* t, long size0, long size1)
{
  return t->nDimension == 2 && t->size[0] == size0 && t->size[1] == size1;
}

static bool checkSize1D(THCTensor* t, long size0)
{
  return t->nDimension == 1 && t->size[0] == size0;
}

static inline void copyCudaFloatingType(THCState *state, THCudaIntTensor *buf, THCTensor *t) {
  #ifdef THC_REAL_IS_FLOAT
  THCudaIntTensor_copyCudaFloat(state, buf, t);
  #elif defined(THC_REAL_IS_DOUBLE)
  THCudaIntTensor_copyCudaDouble(state, buf, t);
  #elif defined(THC_REAL_IS_HALF)
  THCudaIntTensor_copyCudaHalf(state, buf, t);
  #endif
}

void THNN_(SparseLinear_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias)
{
  THAssert(THCTensor_(checkGPU)(state, 4, input, output, weight, bias));

  long h;
  long outDim = THCTensor_(size)(state, weight, 0);
  long inDim = THCTensor_(size)(state, weight, 1);

  THArgCheck(checkInput(input), 2, "input size must be nnz x 3");
  THArgCheck(THCTensor_(nDimension)(state, output) == 2, 3, "output must be batchsize x outputsize");
  THArgCheck(checkSize1D(bias, outDim), 5, "bias size wrong");

  weight = THCTensor_(newContiguous)(state, weight);

  long batchnum = THCTensor_(size)(state, output, 0);
  long nnz = THCTensor_(size)(state, input, 0);

  THCTensor *buffer = THCTensor_(new)(state);
  THCTensor *sel = THCTensor_(new)(state);
  THCTensor *values = THCTensor_(new)(state);
  THCudaIntTensor *rowbuf = THCudaIntTensor_new(state);
  THCudaIntTensor *csrPtrs = THCudaIntTensor_new(state);
  THCudaIntTensor *colInds = THCudaIntTensor_new(state);

  THCTensor_(resize1d)(state, values, nnz);
  THCudaIntTensor_resize1d(state, rowbuf, nnz);
  THCudaIntTensor_resize1d(state, colInds, nnz);
  THCudaIntTensor_resize1d(state, csrPtrs, batchnum+1);

  // Get data ready for cusparse, need CudaInt buffers
  // We do not need to sort, since rows are already in order
  // If rows might get out of order in future implementations, or if cusparse
  //    complains with an illegal memory access, sort like we do in AccGradParameters
  THCTensor_(select)(state, sel, input, 1, 0);
  copyCudaFloatingType(state, rowbuf, sel);
  THCTensor_(select)(state, sel, input, 1, 1);
  copyCudaFloatingType(state, colInds, sel);
  THCTensor_(select)(state, sel, input, 1, 2);
  THCTensor_(copyCuda)(state, values, sel);

  init_cusparse();
  cusparseXcoo2csr(cusparse_handle,
      THCudaIntTensor_data(state, rowbuf), nnz, batchnum,
      THCudaIntTensor_data(state, csrPtrs), CUSPARSE_INDEX_BASE_ONE);

  // output = bias
  THCTensor_(resize2d)(state, buffer, outDim, batchnum);
  THCTensor_(zero)(state, buffer);
  for (h=0; h<batchnum; h++) {
    THCTensor_(select)(state, sel, buffer, 1, h);
    THCTensor_(copy)(state, sel, bias);
  }

  // output = W * x
  real one = ScalarConvert<int, real>::to(1);

#if CUDA_VERSION >= 11000
  int64_t m = batchnum;
  int64_t n = outDim;
  int64_t k = inDim;
  int64_t nnz64 = nnz;
  #ifdef THC_REAL_IS_FLOAT
  cudaDataType cusparse_value_type = CUDA_R_32F;
  #elif defined(THC_REAL_IS_DOUBLE)
  cudaDataType cusparse_value_type = CUDA_R_64F;
  #elif defined(THC_REAL_IS_HALF)
  cudaDataType cusparse_value_type = CUDA_R_16F;
  #endif

  cusparseSpMatDescr_t descA;
  cusparseCreateCsr(
    &descA,                               /* output */
    m, k, nnz64,                          /* rows, cols, number of non zero elements */
    THCudaIntTensor_data(state, csrPtrs), /* row offsets of the sparse matrix, size = rows +1 */
    THCudaIntTensor_data(state, colInds), /* column indices of the sparse matrix, size = nnz */
    THCTensor_(data)(state, values),      /* values of the sparse matrix, size = nnz */
    CUSPARSE_INDEX_32I,                   /* data type of row offsets index */
    CUSPARSE_INDEX_32I,                   /* data type of col indices */
    CUSPARSE_INDEX_BASE_ZERO,             /* base index of row offset and col indes */
    cusparse_value_type                   /* data type of values */
  );

  cusparseDnMatDescr_t descB;
  cusparseCreateDnMat(
    &descB,                               /* output */
    k, n, inDim,                          /* rows, cols, leading dimension */
    THCTensor_(data)(state, weight),      /* values */
    cusparse_value_type,                  /* data type of values */
    CUSPARSE_ORDER_COL                    /* memory layout, ONLY column-major is supported now */
  );

  cusparseDnMatDescr_t descC;
  cusparseCreateDnMat(
    &descC,                               /* output */
    m, n, batchnum,                       /* rows, cols, leading dimension */
    THCTensor_(data)(state, buffer),      /* values */
    cusparse_value_type,                  /* data type of values */
    CUSPARSE_ORDER_COL                    /* memory layout, ONLY column-major is supported now */
  );

  // cusparseSpMM_bufferSize returns the bufferSize that can be used by cusparseSpMM
  size_t bufferSize;
  cusparseSpMM_bufferSize(
    cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &one,
    descA, descB,
    &one,
    descC,
    cusparse_value_type,    /* data type in which the computation is executed */
    CUSPARSE_SPMM_CSR_ALG1, /* default computing algorithm for CSR sparse matrix format */
    &bufferSize             /* output */
  );

  void *external_buffer;
  THCudaMalloc(state, &external_buffer, bufferSize);

  cusparseSpMM(
    cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &one,
    descA, descB,
    &one,
    descC,
    cusparse_value_type,    /* data type in which the computation is executed */
    CUSPARSE_SPMM_CSR_ALG1, /* default computing algorithm for CSR sparse matrix format */
    external_buffer         /* external buffer */
  );

  cusparseDestroySpMat(descA);
  cusparseDestroyDnMat(descB);
  cusparseDestroyDnMat(descC);
  THCudaFree(state, external_buffer);
#else
  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ONE);
  #ifdef THC_REAL_IS_FLOAT
  cusparseScsrmm(cusparse_handle,
  #elif defined(THC_REAL_IS_DOUBLE)
  cusparseDcsrmm(cusparse_handle,
  #endif
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      batchnum, outDim, inDim, nnz,
      &one,
      descr,
      THCTensor_(data)(state, values),
      THCudaIntTensor_data(state, csrPtrs),
      THCudaIntTensor_data(state, colInds),
      THCTensor_(data)(state, weight), inDim,
      &one, THCTensor_(data)(state, buffer), batchnum
  );
  THCTensor_(transpose)(state, buffer, NULL, 0, 1);

  cusparseDestroyMatDescr(descr);
#endif
  // We do work in the buffer to keep the output contiguous
  THCTensor_(copy)(state, output, buffer);

  THCTensor_(free)(state, buffer);
  THCTensor_(free)(state, sel);
  THCTensor_(free)(state, values);
  THCTensor_(free)(state, weight);
  THCudaIntTensor_free(state, rowbuf);
  THCudaIntTensor_free(state, colInds);
  THCudaIntTensor_free(state, csrPtrs);
}

void THNN_(SparseLinear_accGradParameters)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *weight,
           THCTensor *bias,
           accreal weightDecay,
           accreal scale)
{
  long outDim = THCTensor_(size)(state, weight, 0);
  long inDim = THCTensor_(size)(state, weight, 1);

  THArgCheck(checkInput(input), 2, "input size must be batchsize x nnz x 2");
  THArgCheck(checkSize2D(gradWeight, outDim, inDim), 4, "gradWeight size wrong");
  THArgCheck(checkSize1D(gradBias, outDim), 5, "gradBias size wrong");

  weight = THCTensor_(newContiguous)(state, weight);
  long nnz = THCTensor_(size)(state, input, 0);
  long batchnum = THCTensor_(size)(state, gradOutput, 0);

  THCTensor *buf = THCTensor_(new)(state);
  THCTensor *cols = THCTensor_(new)(state);
  THCTensor *sel = THCTensor_(new)(state);
  THCudaLongTensor *inds = THCudaLongTensor_new(state);
  THCTensor *values = THCTensor_(new)(state);
  THCudaIntTensor *colbuf = THCudaIntTensor_new(state);
  THCudaIntTensor *colPtrs = THCudaIntTensor_new(state);
  THCudaIntTensor *rowInds = THCudaIntTensor_new(state);

  THCTensor_(select)(state, sel, input, 1, 0); // rowInds
  THCTensor_(select)(state, cols, input, 1, 1); // colInds
  THCTensor_(cadd)(state, buf, sel, batchnum, cols); // colInds * buatchdim + rowInds
  THCTensor_(sort)(state, buf, inds, buf, 0, 0); // Indicies are now in ind
  THCTensor_(indexSelect)(state, buf, input, 0, inds);

  THCTensor_(resize1d)(state, values, nnz);
  THCudaIntTensor_resize1d(state, colbuf, nnz);
  THCudaIntTensor_resize1d(state, rowInds, nnz);
  THCudaIntTensor_resize1d(state, colPtrs, inDim+1);

  // Get data ready for cusparse, need CudaInt buffers
  THCTensor_(select)(state, sel, buf, 1, 0);
  copyCudaFloatingType(state, rowInds, sel);
  THCTensor_(select)(state, sel, buf, 1, 1);
  copyCudaFloatingType(state, colbuf, sel);
  THCTensor_(select)(state, sel, buf, 1, 2);
  THCTensor_(copyCuda)(state, values, sel);

  init_cusparse();
  // Secretly coo2csc
  cusparseXcoo2csr(cusparse_handle,
      THCudaIntTensor_data(state, colbuf), nnz, inDim,
      THCudaIntTensor_data(state, colPtrs), CUSPARSE_INDEX_BASE_ONE);

  // FORTRAN expects contiguous col-major matricies
  THCTensor *tgradOutput = THCTensor_(new)(state);
  THCTensor_(transpose)(state, tgradOutput, gradOutput, 0, 1);
  THCTensor_(resize2d)(state, buf, batchnum, outDim);
  THCTensor_(copy)(state, buf, tgradOutput);
  THCTensor_(free)(state, tgradOutput);

  real one = ScalarConvert<int, real>::to(1);
  
#if CUDA_VERSION >= 11000
  int64_t m = inDim;
  int64_t n = outDim;
  int64_t k = batchnum;
  int64_t nnz64 = nnz;
  #ifdef THC_REAL_IS_FLOAT
  cudaDataType cusparse_value_type = CUDA_R_32F;
  #elif defined(THC_REAL_IS_DOUBLE)
  cudaDataType cusparse_value_type = CUDA_R_64F;
  #elif defined(THC_REAL_IS_HALF)
  cudaDataType cusparse_value_type = CUDA_R_16F;
  #endif

  cusparseSpMatDescr_t descA;
  cusparseCreateCsr(
    &descA,                               /* output */
    m, k, nnz64,                          /* rows, cols, number of non zero elements */
    THCudaIntTensor_data(state, colPtrs), /* row offsets of the sparse matrix, size = rows +1 */
    THCudaIntTensor_data(state, rowInds), /* column indices of the sparse matrix, size = nnz */
    THCTensor_(data)(state, values),      /* values of the sparse matrix, size = nnz */
    CUSPARSE_INDEX_32I,                   /* data type of row offsets index */
    CUSPARSE_INDEX_32I,                   /* data type of col indices */
    CUSPARSE_INDEX_BASE_ZERO,             /* base index of row offset and col indes */
    cusparse_value_type                   /* data type of values */
  );

  cusparseDnMatDescr_t descB;
  cusparseCreateDnMat(
    &descB,                               /* output */
    k, n, batchnum,                          /* rows, cols, leading dimension */
    THCTensor_(data)(state, buf),         /* values */
    cusparse_value_type,                  /* data type of values */
    CUSPARSE_ORDER_COL                    /* memory layout, ONLY column-major is supported now */
  );

  cusparseDnMatDescr_t descC;
  cusparseCreateDnMat(
    &descC,                               /* output */
    m, n, inDim,                       /* rows, cols, leading dimension */
    THCTensor_(data)(state, gradWeight),         /* values */
    cusparse_value_type,                  /* data type of values */
    CUSPARSE_ORDER_COL                    /* memory layout, ONLY column-major is supported now */
  );

  // cusparseSpMM_bufferSize returns the bufferSize that can be used by cusparseSpMM
  size_t bufferSize;
  cusparseSpMM_bufferSize(
    cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &one,
    descA, descB,
    &one,
    descC,
    cusparse_value_type,    /* data type in which the computation is executed */
    CUSPARSE_SPMM_CSR_ALG1, /* default computing algorithm for CSR sparse matrix format */
    &bufferSize             /* output */
  );

  void *external_buffer;
  THCudaMalloc(state, &external_buffer, bufferSize);

  cusparseSpMM(
    cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &one,
    descA, descB,
    &one,
    descC,
    cusparse_value_type,    /* data type in which the computation is executed */
    CUSPARSE_SPMM_CSR_ALG1, /* default computing algorithm for CSR sparse matrix format */
    external_buffer         /* external buffer */
  );

  cusparseDestroySpMat(descA);
  cusparseDestroyDnMat(descB);
  cusparseDestroyDnMat(descC);
  THCudaFree(state, external_buffer);
#else
  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ONE);
  #ifdef THC_REAL_IS_FLOAT
  cusparseScsrmm(cusparse_handle,
  #elif defined(THC_REAL_IS_DOUBLE)
  cusparseDcsrmm(cusparse_handle,
  #endif
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      inDim, outDim, batchnum, nnz,
      &one,
      descr,
      THCTensor_(data)(state, values),
      THCudaIntTensor_data(state, colPtrs),
      THCudaIntTensor_data(state, rowInds),
      THCTensor_(data)(state, buf), batchnum,
      &one, THCTensor_(data)(state, gradWeight), inDim
  );
  cusparseDestroyMatDescr(descr);
#endif

  THCTensor_(sum)(state, buf, gradOutput, 0, 1);
  THCTensor_(resize1d)(state, buf, outDim);
  THCTensor_(cadd)(state, gradBias, gradBias, scale, buf);

  if (weightDecay != 0)
  {
    THCTensor_(cadd)(state, gradWeight, gradWeight, weightDecay, weight);
    THCTensor_(cadd)(state, gradBias, gradBias, weightDecay, bias);
  }

  THCTensor_(free)(state, weight);
  THCTensor_(free)(state, buf);
  THCTensor_(free)(state, sel);
  THCTensor_(free)(state, cols);
  THCudaLongTensor_free(state, inds);
  THCTensor_(free)(state, values);
  THCudaIntTensor_free(state, colbuf);
  THCudaIntTensor_free(state, rowInds);
  THCudaIntTensor_free(state, colPtrs);
}

void THNN_(SparseLinear_legacyUpdateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias) {
  THError("CUDA does not support legacy input format, please use a table of nnz x 2 vectors");
}
void THNN_(SparseLinear_legacyAccGradParameters)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *weight,
           THCTensor *bias,
           accreal weightDecay,
           accreal scale) {
  THError("CUDA does not support legacy input format, please use a table of nnz x 2 vectors");
}

// Dense updates are pretty fast on the GPU
void THNN_(SparseLinear_zeroGradParameters)(
           THCState *state,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *lastInput) {
  THCTensor_(zero)(state, gradWeight);
  THCTensor_(zero)(state, gradBias);
}

void THNN_(SparseLinear_updateParameters)(
           THCState *state,
           THCTensor *weight,
           THCTensor *bias,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *lastInput,
           accreal learningRate) {
  THCTensor_(cadd)(state, weight, weight, -learningRate, gradWeight);
  THCTensor_(cadd)(state, bias, bias, -learningRate, gradBias);
}

#endif
