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

#pragma once

#include "numpy.h"
#include "core.h"
#include "deserializer.h"

namespace legate {
namespace numpy {

struct DiagArgs {
  bool extract;
  bool needs_reduction;
  int32_t k;
  Shape shape;
  Array out;
  Array in;
};

void deserialize(Deserializer& ctx, DiagArgs& args);

class DiagTask : public NumPyTask<DiagTask> {
 public:
  static const int TASK_ID = NUMPY_DIAG;
  static const int REGIONS = 0;

 public:
  static void cpu_variant(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& regions,
                          Legion::Context ctx,
                          Legion::Runtime* runtime);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& regions,
                          Legion::Context ctx,
                          Legion::Runtime* runtime);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& regions,
                          Legion::Context ctx,
                          Legion::Runtime* runtime);
#endif
};

}  // namespace numpy
}  // namespace legate