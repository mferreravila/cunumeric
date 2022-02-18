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

#include "cunumeric/pitches.h"
#include "cunumeric/fft/fft_util.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, fftType FFT_TYPE, LegateTypeCode CODE_OUT, LegateTypeCode CODE_IN, int DIM>
struct FFTImplBody;

template <VariantKind KIND, fftType FFT_TYPE>
struct FFTImpl {

  template <LegateTypeCode CODE_IN, int DIM, std::enable_if_t<( (DIM <= 3) && FFT<FFT_TYPE,CODE_IN>::valid) >* = nullptr>
  void operator()(FFTArgs& args) const
  {
    using INPUT_TYPE   = legate_type_of<CODE_IN>;
    using OUTPUT_TYPE  = legate_type_of<FFT<FFT_TYPE,CODE_IN>::CODE_OUT>;

    auto in_rect  = args.input.shape<DIM>();
    auto out_rect = args.output.shape<DIM>();
    if (in_rect.empty() || out_rect.empty()) return;

    auto input  = args.input.read_accessor<INPUT_TYPE, DIM>(in_rect);
    auto output = args.output.write_accessor<OUTPUT_TYPE, DIM>(out_rect);

    FFTImplBody<KIND, FFT_TYPE, FFT<FFT_TYPE,CODE_IN>::CODE_OUT, CODE_IN, DIM>()(output, input, out_rect, in_rect, args.direction);
  }

  // We only support up to 3D FFTs for now
  template <LegateTypeCode CODE_IN, int DIM, std::enable_if_t<( (DIM > 3) || !FFT<FFT_TYPE,CODE_IN>::valid )>* = nullptr>
  void operator()(FFTArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct FFTDispatch {
  template <fftType FFT_TYPE>
  void operator()(FFTArgs& args) const
  {
    // Not expecting changing dimensions, at least for now
    assert(args.input.dim() == args.output.dim());

    double_dispatch(args.input.dim(), args.input.code(), FFTImpl<KIND, FFT_TYPE>{}, args);
  }
};

template <VariantKind KIND>
static void fft_template(TaskContext& context)
{
  FFTArgs args;

  auto& inputs   = context.inputs();
  auto& outputs  = context.outputs();
  auto& scalars  = context.scalars();


  args.output    = std::move(outputs[0]);
  args.input     = std::move(inputs[0]);
  args.type      = scalars[0].value<fftType>();
  args.direction = scalars[1].value<fftDirection>();

  fft_dispatch(args.type, FFTDispatch<KIND>{}, args);
}

}  // namespace cunumeric
