/* Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "cunumeric/fft/fft.h"
#include "cunumeric/fft/fft_template.inl"

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <int DIM>
__host__ static inline void cufft_operation(void* output,
                                            void* input,
                                            const Rect<DIM>& out_rect,
                                            const Rect<DIM>& in_rect,
                                            std::vector<int64_t>& axes,
                                            fftType type,
                                            fftDirection direction)
{
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    size_t workarea_size;
    size_t num_elements;
    int n[DIM];
    int inembed[DIM];
    int onembed[DIM];

    const Point<DIM> zero   = Point<DIM>::ZEROES();
    const Point<DIM> one    = Point<DIM>::ONES();
    Point<DIM> fft_size_in  =  in_rect.hi -  in_rect.lo + one;
    Point<DIM> fft_size_out = out_rect.hi - out_rect.lo + one;
    num_elements = 1;
    for(int i = 0; i < DIM; ++i) {
      n[i]          = (type == fftType::FFT_R2C || type == fftType::FFT_D2Z) ? fft_size_in[i] : fft_size_out[i];
      inembed[i]    = fft_size_in[i];
      onembed[i]    = fft_size_out[i];
      num_elements *= n[i];
    }

    // Create the plan
    cufftHandle plan;
    CHECK_CUFFT(cufftCreate(&plan));
    CHECK_CUFFT(cufftSetAutoAllocation(plan, 0 /*we'll do the allocation*/));
    CHECK_CUFFT(cufftSetStream(plan, stream));

    // Create the plan and allocate a temporary buffer for it if it needs one
    CHECK_CUFFT(cufftMakePlanMany(plan, DIM, n, inembed, 1, 1, onembed, 1, 1, (cufftType)type, 1, &workarea_size));

    DeferredBuffer<uint8_t, 1> workarea_buffer;
    if(workarea_size > 0) {
      const Point<1> zero1d(0);
      workarea_buffer =
        DeferredBuffer<uint8_t, 1>(Rect<1>(zero1d, Point<1>(workarea_size - 1)),
                                   Memory::GPU_FB_MEM,
                                   nullptr /*initial*/,
                                   128 /*alignment*/);
      void* workarea = workarea_buffer.ptr(zero1d);
      CHECK_CUFFT(cufftSetWorkArea(plan, workarea));
    }

    // FFT the input data
    CHECK_CUFFT(cufftXtExec(plan, input, output, (int)direction));

    // Clean up our resources, DeferredBuffers are cleaned up by Legion
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaStreamDestroy(stream));
}

template<int DIM, typename OUTPUT_TYPE, typename INPUT_TYPE>
struct cufft_axes_plan{
  __host__ static inline void execute(cufftHandle plan,
                                      AccessorWO<OUTPUT_TYPE, DIM> out,
                                      AccessorRO<INPUT_TYPE, DIM> in,
                                      const Rect<DIM>& out_rect,
                                      const Rect<DIM>& in_rect,
                                      int axis,
                                      fftDirection direction) {
      const Point<DIM> zero   = Point<DIM>::ZEROES();
      CHECK_CUFFT(cufftXtExec(plan, (void*)in.ptr(zero), (void*)out.ptr(zero), (int)direction));
  }
};

template<typename OUTPUT_TYPE, typename INPUT_TYPE>
struct cufft_axes_plan<3, OUTPUT_TYPE, INPUT_TYPE>{
  __host__ static inline void execute(cufftHandle plan,
                                      AccessorWO<OUTPUT_TYPE, 3> out,
                                      AccessorRO<INPUT_TYPE,  3> in,
                                      const Rect<3>& out_rect,
                                      const Rect<3>& in_rect,
                                      int axis,
                                      fftDirection direction) {
    bool is_inner_axis = (axis == 1);
    if(is_inner_axis) {
      // TODO: use PointInRectIterator<DIM>
      auto num_slices = in_rect.hi[0] - in_rect.lo[0] + 1;
      for(unsigned n = 0; n < num_slices; ++n){
        const Point<3> offset = Point<3>(n, 0, 0);
        CHECK_CUFFT(cufftXtExec(plan, (void*)in.ptr(offset), (void*)out.ptr(offset), (int)direction));
      }
    }
    else {
      const Point<3> zero   = Point<3>::ZEROES();
      CHECK_CUFFT(cufftXtExec(plan, (void*)in.ptr(zero), (void*)out.ptr(zero), (int)direction));
    }
  }
};

// Perform the FFT operation as multiple 1D FFTs along the specified axes.
// For now, it only supports up to 3D FFTs, but final plan is having support for
// N-dimensional FFTs using this approach.
template <int DIM, typename OUTPUT_TYPE, typename INPUT_TYPE>
__host__ static inline void cufft_operation_by_axes(AccessorWO<OUTPUT_TYPE, DIM> out,
                                                    AccessorRO<INPUT_TYPE, DIM> in,
                                                    const Rect<DIM>& out_rect,
                                                    const Rect<DIM>& in_rect,
                                                    std::vector<int64_t>& axes,
                                                    fftType type,
                                                    fftDirection direction)
{
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    size_t workarea_size;
    size_t num_elements;
    int n[DIM];
    int inembed[DIM];
    int onembed[DIM];

    const Point<DIM> zero   = Point<DIM>::ZEROES();
    const Point<DIM> one    = Point<DIM>::ONES();
    Point<DIM> fft_size_in  =  in_rect.hi -  in_rect.lo + one;
    Point<DIM> fft_size_out = out_rect.hi - out_rect.lo + one;
    num_elements = 1;
    for(int i = 0; i < DIM; ++i) {
      n[i]          = fft_size_out[i];
      inembed[i]    = fft_size_in[i];
      onembed[i]    = fft_size_out[i];
      num_elements *= n[i];
    }

    for(auto ax = axes.begin(); ax < axes.end(); ++ax) {
      // Create the plan
      cufftHandle plan;
      CHECK_CUFFT(cufftCreate(&plan));
      CHECK_CUFFT(cufftSetAutoAllocation(plan, 0 /*we'll do the allocation*/));
      CHECK_CUFFT(cufftSetStream(plan, stream));

      // Create the plan and allocate a temporary buffer for it if it needs one
      // For now, contiguous plan with a single batch
      int size_1d = n[*ax];
      // TODO: batches only correct for DIM <= 3. Fix for N-DIM case
      int batches = (DIM == 3 && *ax == 1) ? n[2] : num_elements / n[*ax];
      int istride = 1;
      int ostride = 1;
      for(int i = *ax+1; i < DIM; ++i) {
        istride *= n[i];
        ostride *= n[i];
      }
      int idist = (*ax == DIM-1) ? n[*ax] : 1;
      int odist = (*ax == DIM-1) ? n[*ax] : 1;

      CHECK_CUFFT(cufftMakePlanMany(plan, 1, &size_1d, inembed, istride, idist, onembed, ostride, odist, (cufftType)type, batches, &workarea_size));

      DeferredBuffer<uint8_t, 1> workarea_buffer;
      if(workarea_size > 0) {
        const Point<1> zero1d(0);
        workarea_buffer =
          DeferredBuffer<uint8_t, 1>(Rect<1>(zero1d, Point<1>(workarea_size - 1)),
                                     Memory::GPU_FB_MEM,
                                     nullptr /*initial*/,
                                     128 /*alignment*/);
        void* workarea = workarea_buffer.ptr(zero1d);
        CHECK_CUFFT(cufftSetWorkArea(plan, workarea));
      }

      // For dimensions higher than 2D, we need to iterate through the input volume as 2D slices due to
      // limitations of cuFFT indexing in 1D
      // TODO: following function only correct for DIM <= 3. Fix for N-DIM case
      cufft_axes_plan<DIM, OUTPUT_TYPE, INPUT_TYPE>::execute(plan, out, in, out_rect, in_rect, *ax, direction);

      // Clean up our resources, DeferredBuffers are cleaned up by Legion
      CHECK_CUFFT(cufftDestroy(plan));
      CHECK_CUDA(cudaStreamDestroy(stream));
    }
}


template <fftType FFT_TYPE, LegateTypeCode CODE_OUT, LegateTypeCode CODE_IN, int DIM>
struct FFTImplBody<VariantKind::GPU, FFT_TYPE, CODE_OUT, CODE_IN, DIM> {
  using INPUT_TYPE  = legate_type_of<CODE_IN>;
  using OUTPUT_TYPE = legate_type_of<CODE_OUT>;

  __host__ void operator()(AccessorWO<OUTPUT_TYPE, DIM> out,
                           AccessorRO<INPUT_TYPE, DIM> in,
                           const Rect<DIM>& out_rect,
                           const Rect<DIM>& in_rect,
                           std::vector<int64_t>& axes,
                           fftDirection direction) const
  {
    const Point<DIM> zero = Point<DIM>::ZEROES();
    void* out_ptr = (void*) out.ptr(zero);
    void* in_ptr  = (void*) in.ptr(zero);

    if(axes.size() > 0) {
      // FFTs are computed as 1D over different axes. Slower than performing the full FFT in a single step
      cufft_operation_by_axes<DIM, OUTPUT_TYPE, INPUT_TYPE>(out, in, out_rect, in_rect, axes, FFT_TYPE, direction);
    }
    else {
      // FFTs are computed as a single step of DIM
      cufft_operation<DIM>(out_ptr, in_ptr, out_rect, in_rect, axes, FFT_TYPE, direction);      
    }
  }
};

/*static*/ void FFTTask::gpu_variant(TaskContext& context)
{
  fft_template<VariantKind::GPU>(context);
};

}  // namespace cunumeric
