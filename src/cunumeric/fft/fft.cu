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
                                            fftType type,
                                            fftDirection direction)
{
    // cufft plans are awful things that call cudaMalloc/cudaFree which
    // completely destroys asynchronous execution so we need to cache
    // these plans to avoid calling it as often as possible
    // constexpr size_t MAX_PLANS = 4;
    // struct cufftPlan {
    //  public:
    //   cufftPlan(void) : fftshape(Point<DIM>::ZEROES()) {}

    //  public:
    //   cufftHandle forward;
    //   cufftHandle backward;
    //   Point<DIM> fftshape;
    //   size_t workarea_size;
    //   unsigned lru_index;
    // };
    // static cufftPlan cufft_plan_cache[LEGION_MAX_NUM_PROCS][MAX_PLANS];

    // Instead of doing the large tile case, we can instead do this
    // by transforming both the input and the filter to the frequency
    // domain using an FFT, perform the convolution with a point-wise
    // multiplication, and then transform the result back to the spatial domain
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Create the plan for going in both directions
    cufftHandle plan;
    CHECK_CUFFT(cufftCreate(&plan));
    CHECK_CUFFT(cufftSetAutoAllocation(plan, 0 /*we'll do the allocation*/));
    CHECK_CUFFT(cufftSetStream(plan, stream));

    size_t workarea_size;
    int n[DIM];
    int inembed[DIM];
    int onembed[DIM];

    const Point<DIM> zero   = Point<DIM>::ZEROES();
    const Point<DIM> one    = Point<DIM>::ONES();
    Point<DIM> fft_size_in  =  in_rect.hi -  in_rect.lo + one;
    Point<DIM> fft_size_out = out_rect.hi - out_rect.lo + one;
    for(int i = 0; i < DIM; ++i) {
      n[i]       = fft_size_out[i];
      inembed[i] = fft_size_in[i];
      onembed[i] = fft_size_out[i];
      printf("Size (outermost to innermost) %d\n", n[i]);
    }

    // DIM = 2 n[0] 512 n[1] 1024
    // inembed[1] = n[1]

    // DIM = 3 n[0] 256 n[1] 512  n[2] 1024
    // inembed[2] = n[2]
    // inembed[1] = n[1]

    // Create the plan and allocate a temporary buffer for it if it needs one
    // For now, contiguous plan with a single batch
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

    // double in_host[256];
    // CHECK_CUDA(cudaMemcpy(in_host, input, sizeof(double) * 256, cudaMemcpyDeviceToHost));
    // printf("[0] %f [1] %f [2] %f [3] %f\n", in_host[0], in_host[1], in_host[2], in_host[3]);

    // FFT the input data
    CHECK_CUFFT(cufftXtExec(plan, input, output, (int)direction));

    // Copy the result data out of the temporary buffer and scale
    // because CUFFT inverse does not perform the scale for us
    // pitch = 1;
    // FFTPitches<DIM> fft_pitches;
    // for (int d = DIM - 1; d >= 0; d--) {
    //   fft_pitches[d] = pitch;
    //   pitch *= fftsize[d];
    // }
    // const OUTPUT scaling_factor = OUTPUT(1) / pitch;
    // Point<DIM> buffer_offset;
    // for (int d = 0; d < DIM; d++)
    //   buffer_offset[d] =
    //     centers[d] - (((extents[d] % 2) == 0) ? 1 : 0) +
    //     ((offset_bounds.lo[d] < out_rect.lo[d]) ? (subrect.lo[d] - out_rect.lo[d]) : centers[d]);
    // Point<DIM> output_bounds = subrect.hi - subrect.lo + one;
    // pitch                    = 1;
    // for (int d = DIM - 1; d >= 0; d--) {
    //   copy_pitches[d] = FastDivmodU64(pitch);
    //   pitch *= output_bounds[d];
    // }
    // blocks = (pitch + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // copy_from_buffer<OUTPUT, DIM><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
    //   filter_ptr, out, buffer_offset, subrect.lo, copy_pitches, fft_pitches, pitch, scaling_factor);

#if 0
    // This is useful debugging code for finding the output
    OUTPUT *buffer = (OUTPUT*)malloc(buffervolume*sizeof(OUTPUT));
    CHECK_CUDA( cudaMemcpyAsync(buffer, filter_ptr, buffervolume*sizeof(OUTPUT), cudaMemcpyDeviceToHost, stream) );
    CHECK_CUDA( cudaStreamSynchronize(stream) );
    for (unsigned idx = 0; idx < buffervolume; idx++) {
      if ((idx % fftsize[DIM-1]) == 0)
        printf("\n");
      printf("%.8g ", buffer[idx]*scaling_factor);
    }
    printf("\n");
    free(buffer);
#endif
    // Clean up our resources, DeferredBuffers are cleaned up by Legion
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaStreamDestroy(stream));
}


template <fftType FFT_TYPE, LegateTypeCode CODE_OUT, LegateTypeCode CODE_IN, int DIM>
struct FFTImplBody<VariantKind::GPU, FFT_TYPE, CODE_OUT, CODE_IN, DIM> {
  using INPUT_TYPE  = legate_type_of<CODE_IN>;
  using OUTPUT_TYPE = legate_type_of<CODE_OUT>;

  __host__ void operator()(AccessorWO<OUTPUT_TYPE, DIM> out,
                           AccessorRO<INPUT_TYPE, DIM> in,
                           const Rect<DIM>& out_rect,
                           const Rect<DIM>& in_rect,
                           fftDirection direction) const
  {
    const Point<DIM> zero = Point<DIM>::ZEROES();
    void* out_ptr = (void*) out.ptr(zero);
    void* in_ptr  = (void*) in.ptr(zero);
    cufft_operation<DIM>(out_ptr, in_ptr, out_rect, in_rect, FFT_TYPE, direction);
  }
};

/*static*/ void FFTTask::gpu_variant(TaskContext& context)
{
  fft_template<VariantKind::GPU>(context);
};

}  // namespace cunumeric
