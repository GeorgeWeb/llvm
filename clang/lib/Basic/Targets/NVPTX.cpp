//===--- NVPTX.cpp - Implement NVPTX target feature support ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements NVPTX TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "Targets.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace clang::targets;

static constexpr Builtin::Info BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  {#ID, TYPE, ATTRS, nullptr, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
#define LIBBUILTIN(ID, TYPE, ATTRS, HEADER)                                    \
  {#ID, TYPE, ATTRS, nullptr, HeaderDesc::HEADER, ALL_LANGUAGES},
#define TARGET_BUILTIN(ID, TYPE, ATTRS, FEATURE)                               \
  {#ID, TYPE, ATTRS, FEATURE, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
#include "clang/Basic/BuiltinsNVPTX.def"
};

const char *const NVPTXTargetInfo::GCCRegNames[] = {"r0"};

NVPTXTargetInfo::NVPTXTargetInfo(const llvm::Triple &Triple,
                                 const TargetOptions &Opts,
                                 unsigned TargetPointerWidth)
    : TargetInfo(Triple) {
  assert((TargetPointerWidth == 32 || TargetPointerWidth == 64) &&
         "NVPTX only supports 32- and 64-bit modes.");

  PTXVersion = 32;
  for (const StringRef Feature : Opts.FeaturesAsWritten) {
    int PTXV;
    if (!Feature.starts_with("+ptx") ||
        Feature.drop_front(4).getAsInteger(10, PTXV))
      continue;
    PTXVersion = PTXV; // TODO: should it be max(PTXVersion, PTXV)?
  }

  TLSSupported = false;
  VLASupported = false;
  AddrSpaceMap = &NVPTXAddrSpaceMap;
  UseAddrSpaceMapMangling = true;
  HasLegalHalfType = true;
  HasFloat16 = true;
  // __bf16 is always available as a load/store only type.
  BFloat16Width = BFloat16Align = 16;
  BFloat16Format = &llvm::APFloat::BFloat();

  // Define available target features
  // These must be defined in sorted order!
  NoAsmVariants = true;
  GPU = CudaArch::SM_20;

  if (TargetPointerWidth == 32)
    resetDataLayout("e-p:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64");
  else if (Opts.NVPTXUseShortPointers)
    resetDataLayout(
        "e-p3:32:32-p4:32:32-p5:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64");
  else
    resetDataLayout("e-i64:64-i128:128-v16:16-v32:32-n16:32:64");

  // If possible, get a TargetInfo for our host triple, so we can match its
  // types.
  llvm::Triple HostTriple(Opts.HostTriple);
  if (!HostTriple.isNVPTX())
    HostTarget = AllocateTarget(llvm::Triple(Opts.HostTriple), Opts);

  // If no host target, make some guesses about the data layout and return.
  if (!HostTarget) {
    LongWidth = LongAlign = TargetPointerWidth;
    PointerWidth = PointerAlign = TargetPointerWidth;
    switch (TargetPointerWidth) {
    case 32:
      SizeType = TargetInfo::UnsignedInt;
      PtrDiffType = TargetInfo::SignedInt;
      IntPtrType = TargetInfo::SignedInt;
      break;
    case 64:
      SizeType = TargetInfo::UnsignedLong;
      PtrDiffType = TargetInfo::SignedLong;
      IntPtrType = TargetInfo::SignedLong;
      break;
    default:
      llvm_unreachable("TargetPointerWidth must be 32 or 64");
    }

    MaxAtomicInlineWidth = TargetPointerWidth;
    return;
  }

  // Copy properties from host target.
  PointerWidth = HostTarget->getPointerWidth(LangAS::Default);
  PointerAlign = HostTarget->getPointerAlign(LangAS::Default);
  BoolWidth = HostTarget->getBoolWidth();
  BoolAlign = HostTarget->getBoolAlign();
  IntWidth = HostTarget->getIntWidth();
  IntAlign = HostTarget->getIntAlign();
  HalfWidth = HostTarget->getHalfWidth();
  HalfAlign = HostTarget->getHalfAlign();
  FloatWidth = HostTarget->getFloatWidth();
  FloatAlign = HostTarget->getFloatAlign();
  DoubleWidth = HostTarget->getDoubleWidth();
  DoubleAlign = HostTarget->getDoubleAlign();
  LongWidth = HostTarget->getLongWidth();
  LongAlign = HostTarget->getLongAlign();
  LongLongWidth = HostTarget->getLongLongWidth();
  LongLongAlign = HostTarget->getLongLongAlign();
  MinGlobalAlign = HostTarget->getMinGlobalAlign(/* TypeSize = */ 0);
  NewAlign = HostTarget->getNewAlign();
  DefaultAlignForAttributeAligned =
      HostTarget->getDefaultAlignForAttributeAligned();
  SizeType = HostTarget->getSizeType();
  IntMaxType = HostTarget->getIntMaxType();
  PtrDiffType = HostTarget->getPtrDiffType(LangAS::Default);
  IntPtrType = HostTarget->getIntPtrType();
  WCharType = HostTarget->getWCharType();
  WIntType = HostTarget->getWIntType();
  Char16Type = HostTarget->getChar16Type();
  Char32Type = HostTarget->getChar32Type();
  Int64Type = HostTarget->getInt64Type();
  SigAtomicType = HostTarget->getSigAtomicType();
  ProcessIDType = HostTarget->getProcessIDType();

  UseBitFieldTypeAlignment = HostTarget->useBitFieldTypeAlignment();
  UseZeroLengthBitfieldAlignment = HostTarget->useZeroLengthBitfieldAlignment();
  UseExplicitBitFieldAlignment = HostTarget->useExplicitBitFieldAlignment();
  ZeroLengthBitfieldBoundary = HostTarget->getZeroLengthBitfieldBoundary();

  // This is a bit of a lie, but it controls __GCC_ATOMIC_XXX_LOCK_FREE, and
  // we need those macros to be identical on host and device, because (among
  // other things) they affect which standard library classes are defined, and
  // we need all classes to be defined on both the host and device.
  MaxAtomicInlineWidth = HostTarget->getMaxAtomicInlineWidth();

  // Properties intentionally not copied from host:
  // - LargeArrayMinWidth, LargeArrayAlign: Not visible across the
  //   host/device boundary.
  // - SuitableAlign: Not visible across the host/device boundary, and may
  //   correctly be different on host/device, e.g. if host has wider vector
  //   types than device.
  // - LongDoubleWidth, LongDoubleAlign: nvptx's long double type is the same
  //   as its double type, but that's not necessarily true on the host.
  //   TODO: nvcc emits a warning when using long double on device; we should
  //   do the same.
}

ArrayRef<const char *> NVPTXTargetInfo::getGCCRegNames() const {
  return llvm::ArrayRef(GCCRegNames);
}

bool NVPTXTargetInfo::hasFeature(StringRef Feature) const {
  return llvm::StringSwitch<bool>(Feature)
      .Cases("ptx", "nvptx", true)
      .Default(false);
}

/*
static void populateDeviceTraitsMacros(llvm::TargetInfo* TTI,
                                       MacroBuilder &Builder) const {
  // TODO: if not SYCL then continue;

  const auto &TargetTable = DeviceConfigFile::TargetTable;
  std::map<StringRef, unsigned int> AllDevicesHave;
  std::map<StringRef, bool> AnyDeviceHas;
  bool AnyDeviceHasAnyAspect = false;
  unsigned int ValidTargets = 0;
  for (size_t i = 0; i < Args.size(); ++i) {
    const auto Arg = Args.getArgString(i);
    llvm::outs() << "populate SYCL Device Traits: Arg = " << Arg << '\n';
  }
  const llvm::Triple TargetTriple = getTriple();

  // Try and find the whole triple, if there's no match, remove parts of the
  // triple from the end to find partial matches.
  auto TargetTripleStr = TargetTriple.getTriple();
  llvm::outs() << "TargetTripleStr = " << TargetTripleStr << '\n';
  bool Found = false;
  bool EmptyTriple = false;
  auto TripleIt = TargetTable.end();

  for (auto [key, val] : TargetTable) {
    //llvm::outs() << "TargetTable key = " << key << '\n';
  }

  while (!Found && !EmptyTriple) {
    TripleIt = TargetTable.find(TargetTripleStr);
    Found = (TripleIt != TargetTable.end());
    if (!Found) {
      auto Pos = TargetTripleStr.find_last_of('-');
      EmptyTriple = (Pos == std::string::npos);
      TargetTripleStr =
          EmptyTriple ? TargetTripleStr : TargetTripleStr.substr(0, Pos);
      llvm::outs() << "Not found, TargetTripleStr = " << TargetTripleStr << '\n';
    }
  }
  llvm::outs() << "Final, TargetTripleStr = " << TargetTripleStr << '\n';
  if (Found) {
    assert(TripleIt != TargetTable.end());
    const auto &TargetInfo = (*TripleIt).second;
    ++ValidTargets;
    llvm::outs() << "Found a valid  target\n";
    const auto &AspectList = TargetInfo.aspects;
    const auto &MaySupportOtherAspects = TargetInfo.maySupportOtherAspects;
    if (!AnyDeviceHasAnyAspect)
      AnyDeviceHasAnyAspect = MaySupportOtherAspects;
    for (const auto &aspect : AspectList) {
      // If target has an entry in the config file, the set of aspects
      // supported by all devices supporting the target is 'AspectList'. If
      // there's no entry, such set is empty.
      const auto &AspectIt = AllDevicesHave.find(aspect);
      if (AspectIt != AllDevicesHave.end())
        ++AllDevicesHave[aspect];
      else
        AllDevicesHave[aspect] = 1;
      // If target has an entry in the config file AND
      // 'MaySupportOtherAspects' is false, the set of aspects supported by
      // any device supporting the target is 'AspectList'. If there's no
      // entry OR 'MaySupportOtherAspects' is true, such set contains all
      // the aspects.
      AnyDeviceHas[aspect] = true;
    }
  }

  if (ValidTargets == 0) {
    // If there's no entry for the target in the device config file, the set
    // of aspects supported by any device supporting the target contains all
    // the aspects.
    AnyDeviceHasAnyAspect = true;
  }

  if (AnyDeviceHasAnyAspect) {
    // There exists some target that supports any given aspect.
    SmallString<64> MacroAnyDeviceAnyAspect(
        "-D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1");
    Builder.defineMacro(MacroAnyDeviceAnyAspect);
    SYCLDeviceTraitsMacrosArgs.push_back(
        Args.MakeArgString(MacroAnyDeviceAnyAspect));
  } else {
    // Some of the aspects are not supported at all by any of the targets.
    // Thus, we need to define individual macros for each supported aspect.
    for (const auto &[TargetKey, SupportedTarget] : AnyDeviceHas) {
      assert(SupportedTarget);
      SmallString<64> MacroAnyDevice("-D__SYCL_ANY_DEVICE_HAS_");
      MacroAnyDevice += TargetKey;
      MacroAnyDevice += "__=1";
      SYCLDeviceTraitsMacrosArgs.push_back(Args.MakeArgString(MacroAnyDevice));
    }
  }
  for (const auto &[TargetKey, SupportedTargets] : AllDevicesHave) {
    if (SupportedTargets != ValidTargets)
      continue;
    SmallString<64> MacroAllDevices("-D__SYCL_ALL_DEVICES_HAVE_");
    MacroAllDevices += TargetKey;
    MacroAllDevices += "__=1";
    SYCLDeviceTraitsMacrosArgs.push_back(Args.MakeArgString(MacroAllDevices));
  }
}
*/

void NVPTXTargetInfo::getTargetDefines(const LangOptions &Opts,
                                       MacroBuilder &Builder) const {
  Builder.defineMacro("__PTX__");
  Builder.defineMacro("__NVPTX__");
  if (Opts.CUDAIsDevice || Opts.OpenMPIsTargetDevice || Opts.SYCLIsDevice ||
      !HostTarget) {
    // Set __CUDA_ARCH__ or __SYCL_CUDA_ARCH__ for the GPU specified.
    // The SYCL-specific macro is used to distinguish the SYCL and CUDA APIs.
    std::string CUDAArchCode = [this] {
      switch (GPU) {
      case CudaArch::GFX600:
      case CudaArch::GFX601:
      case CudaArch::GFX602:
      case CudaArch::GFX700:
      case CudaArch::GFX701:
      case CudaArch::GFX702:
      case CudaArch::GFX703:
      case CudaArch::GFX704:
      case CudaArch::GFX705:
      case CudaArch::GFX801:
      case CudaArch::GFX802:
      case CudaArch::GFX803:
      case CudaArch::GFX805:
      case CudaArch::GFX810:
      case CudaArch::GFX900:
      case CudaArch::GFX902:
      case CudaArch::GFX904:
      case CudaArch::GFX906:
      case CudaArch::GFX908:
      case CudaArch::GFX909:
      case CudaArch::GFX90a:
      case CudaArch::GFX90c:
      case CudaArch::GFX940:
      case CudaArch::GFX941:
      case CudaArch::GFX942:
      case CudaArch::GFX1010:
      case CudaArch::GFX1011:
      case CudaArch::GFX1012:
      case CudaArch::GFX1013:
      case CudaArch::GFX1030:
      case CudaArch::GFX1031:
      case CudaArch::GFX1032:
      case CudaArch::GFX1033:
      case CudaArch::GFX1034:
      case CudaArch::GFX1035:
      case CudaArch::GFX1036:
      case CudaArch::GFX1100:
      case CudaArch::GFX1101:
      case CudaArch::GFX1102:
      case CudaArch::GFX1103:
      case CudaArch::GFX1150:
      case CudaArch::GFX1151:
      case CudaArch::GFX1200:
      case CudaArch::GFX1201:
      case CudaArch::Generic:
      case CudaArch::LAST:
        break;
      case CudaArch::UNUSED:
      case CudaArch::UNKNOWN:
        assert(false && "No GPU arch when compiling CUDA device code.");
        return "";
      case CudaArch::SM_20:
        return "200";
      case CudaArch::SM_21:
        return "210";
      case CudaArch::SM_30:
        return "300";
      case CudaArch::SM_32:
        return "320";
      case CudaArch::SM_35:
        return "350";
      case CudaArch::SM_37:
        return "370";
      case CudaArch::SM_50:
        return "500";
      case CudaArch::SM_52:
        return "520";
      case CudaArch::SM_53:
        return "530";
      case CudaArch::SM_60:
        return "600";
      case CudaArch::SM_61:
        return "610";
      case CudaArch::SM_62:
        return "620";
      case CudaArch::SM_70:
        return "700";
      case CudaArch::SM_72:
        return "720";
      case CudaArch::SM_75:
        return "750";
      case CudaArch::SM_80:
        return "800";
      case CudaArch::SM_86:
        return "860";
      case CudaArch::SM_87:
        return "870";
      case CudaArch::SM_89:
        return "890";
      case CudaArch::SM_90:
      case CudaArch::SM_90a:
        return "900";
      }
      llvm_unreachable("unhandled CudaArch");
    }();

    // TODO: This looks like a broken merge from upstream - FIXME:
    // - __CUDA_ARCH_FEAT_SM90_ALL support check should not be in else-if.
    if (Opts.SYCLIsDevice) {
      Builder.defineMacro("__SYCL_CUDA_ARCH__", CUDAArchCode);
    } else if (GPU == CudaArch::SM_90a) {
      Builder.defineMacro("__CUDA_ARCH_FEAT_SM90_ALL", "1");
    } else {
      Builder.defineMacro("__CUDA_ARCH__", CUDAArchCode);
    }
  }
}

ArrayRef<Builtin::Info> NVPTXTargetInfo::getTargetBuiltins() const {
  return llvm::ArrayRef(BuiltinInfo,
                        clang::NVPTX::LastTSBuiltin - Builtin::FirstTSBuiltin);
}
