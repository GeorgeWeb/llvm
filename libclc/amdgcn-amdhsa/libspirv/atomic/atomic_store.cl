//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "atomic_helpers.h"
#include <spirv/spirv.h>
#include <spirv/spirv_types.h>

// By the default the fence synscope implies cross-address space ordering, unless
// the address space ordering MMRAs are set.
_CLC_INLINE void builtin_fence_cross(enum Scope scope, unsigned int order) {
  switch (scope) {
  case CrossDevice:
    BUILTIN_FENCE(order, "")
  case Device:
    BUILTIN_FENCE(order, "agent")
  case Workgroup:
    BUILTIN_FENCE(order, "workgroup")
  case Subgroup:
    BUILTIN_FENCE(order, "wavefront")
  case Invocation:
    BUILTIN_FENCE(order, "singlethread")
  }
}

_CLC_INLINE void builtin_fence_global(enum Scope scope, unsigned int order) {
  switch (scope) {
  case CrossDevice:
    BUILTIN_FENCE_MASKED(order, "", "global")
  case Device:
    BUILTIN_FENCE_MASKED(order, "agent", "global")
  case Workgroup:
    BUILTIN_FENCE_MASKED(order, "workgroup", "global")
  case Subgroup:
    BUILTIN_FENCE_MASKED(order, "wavefront", "global")
  case Invocation:
    BUILTIN_FENCE_MASKED(order, "singlethread", "global")
  }
}

_CLC_INLINE void builtin_fence_local(enum Scope scope, unsigned int order) {
  switch (scope) {
  case CrossDevice:
    BUILTIN_FENCE_MASKED(order, "", "local")
  case Device:
    BUILTIN_FENCE_MASKED(order, "agent", "local")
  case Workgroup:
    BUILTIN_FENCE_MASKED(order, "workgroup", "local")
  case Subgroup:
    BUILTIN_FENCE_MASKED(order, "wavefront", "local")
  case Invocation:
    BUILTIN_FENCE_MASKED(order, "singlethread", "local")
  }
}

#undef BUILTIN_FENCE_MASKED
#undef BUILTIN_FENCE

#define AMDGPU_ATOMIC_STORE_IMPL(TYPE, TYPE_MANGLED, AS, AS_MANGLED, SUB1)                                                           \
  _CLC_DEF void                                                                                                                      \
      _Z19__spirv_AtomicStore##P##AS_MANGLED##TYPE_MANGLED##N5__spv5Scope4FlagENS##SUB1##_19MemorySemanticsMask4FlagE##TYPE_MANGLED( \
          volatile AS TYPE *p, enum Scope scope,                                                                                     \
          enum MemorySemanticsMask semantics, TYPE val) {                                                                            \
    int atomic_scope = 0, memory_order = 0;                                                                                          \
    GET_ATOMIC_SCOPE_AND_ORDER(scope, atomic_scope, semantics, memory_order)                                                         \
    __hip_atomic_store(p, val, memory_order, atomic_scope);                                                                          \
    if (semantics & SequentiallyConsistent) {                                                                                        \
        /* This means that following loads will not see stale data and that any                                                      \
        * following global data read is no older than the local atomic value. */                                                     \
        builtin_fence_global(scope, Acquire);                                                                                        \
    }                                                                                                                                \
    return;                                                                                                                          \
  }

#define AMDGPU_ATOMIC_STORE(TYPE, TYPE_MANGLED)                                \
  AMDGPU_ATOMIC_STORE_IMPL(TYPE, TYPE_MANGLED, global, U3AS1, 1)               \
  AMDGPU_ATOMIC_STORE_IMPL(TYPE, TYPE_MANGLED, local, U3AS3, 1)                \
  AMDGPU_ATOMIC_STORE_IMPL(TYPE, TYPE_MANGLED, , , 0)

AMDGPU_ATOMIC_STORE(int, i)
AMDGPU_ATOMIC_STORE(unsigned int, j)
AMDGPU_ATOMIC_STORE(long, l)
AMDGPU_ATOMIC_STORE(unsigned long, m)
AMDGPU_ATOMIC_STORE(float, f)

// TODO implement for fp64

#undef AMDGPU_ATOMIC
#undef AMDGPU_ATOMIC_IMPL
#undef AMDGPU_ATOMIC_STORE
#undef AMDGPU_ATOMIC_STORE_IMPL
#undef AMDGPU_ARCH_GEQ
#undef AMDGPU_ARCH_BETWEEN
#undef GET_ATOMIC_SCOPE_AND_ORDER
