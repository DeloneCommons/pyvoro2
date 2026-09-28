#ifndef PYVORO2_NATIVE_RUNTIME_HPP
#define PYVORO2_NATIVE_RUNTIME_HPP

// The entry/refusal boundary is integer-only. In particular, neither control
// inspection nor profile reporting may test subnormal arithmetic before the
// incoming exception masks have been checked. This header and the optimized
// dispatch code are part of the externally qualified native source closure.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && defined(_M_X64)
#include <xmmintrin.h>
#endif

#if defined(_MSC_VER)
#define PYVORO2_RUNTIME_NOINLINE __declspec(noinline)
#else
#define PYVORO2_RUNTIME_NOINLINE __attribute__((noinline))
#endif

namespace pyvoro2::native_runtime {
namespace py = pybind11;

struct State {
  const char* adapter = "unsupported";
  bool supported = false;
  bool has_x87 = false;
  bool has_mxcsr = false;
  bool has_arm = false;
  std::uint16_t x87_control = 0;
  std::uint16_t x87_status = 0;
  std::uint32_t mxcsr = 0;
  std::uint64_t fpcr = 0;
  std::uint64_t fpsr = 0;
  bool nearest = false;
  bool subnormal = false;
  bool masked = false;
  bool precision = false;
  const char* refusal = "unsupported execution target adapter";

  bool compatible() const noexcept {
    return supported && nearest && subnormal && masked && precision;
  }
};

// A no-inline boundary also gives effective-build qualification a small,
// identifiable instruction range to inspect. Do not replace FNSTCW/FNSTSW by
// their waiting variants: a pending, unmasked x87 exception must be refusable.
PYVORO2_RUNTIME_NOINLINE inline State inspect() noexcept {
  State state;
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
  state.adapter = "x86_64-x87-sse2";
  state.supported = true;
  state.has_x87 = true;
  state.has_mxcsr = true;
  asm volatile("fnstcw %0" : "=m"(state.x87_control) : : "memory");
  asm volatile("fnstsw %0" : "=m"(state.x87_status) : : "memory");
  asm volatile("stmxcsr %0" : "=m"(state.mxcsr) : : "memory");
  state.nearest = (state.x87_control & 0x0c00u) == 0 &&
                  (state.mxcsr & 0x6000u) == 0;
  state.subnormal = (state.mxcsr & 0x8040u) == 0;
  state.masked = (state.x87_control & 0x003fu) == 0x003fu &&
                 (state.mxcsr & 0x1f80u) == 0x1f80u;
  state.precision = std::numeric_limits<long double>::digits != 64 ||
                     (state.x87_control & 0x0300u) == 0x0300u;
  if ((state.x87_control & 0x0c00u) != 0)
    state.refusal = "x87 rounding is not nearest-even";
  else if ((state.mxcsr & 0x6000u) != 0)
    state.refusal = "MXCSR rounding is not nearest-even";
  else if ((state.mxcsr & 0x8040u) != 0)
    state.refusal = "MXCSR FTZ/DAZ does not preserve subnormals";
  else if ((state.x87_control & 0x003fu) != 0x003fu)
    state.refusal = "x87 exception classes are not all masked";
  else if ((state.mxcsr & 0x1f80u) != 0x1f80u)
    state.refusal = "MXCSR exception classes are not all masked";
  else if (!state.precision)
    state.refusal = "x87 precision is not PC64 for declared long double";
#elif defined(_MSC_VER) && defined(_M_X64)
  // MSVC x64 evaluates its binary64 long double in SSE2. Its reviewed native
  // route does not execute proof-sensitive x87 arithmetic. __control87_2 is
  // unavailable on x64; a merged CRT control word is not a raw MXCSR read.
  state.adapter = "msvc-x64-sse2";
  state.supported = std::numeric_limits<long double>::digits == 53;
  state.has_mxcsr = true;
  state.mxcsr = _mm_getcsr();
  state.nearest = (state.mxcsr & 0x6000u) == 0;
  state.subnormal = (state.mxcsr & 0x8040u) == 0;
  state.masked = (state.mxcsr & 0x1f80u) == 0x1f80u;
  state.precision = state.supported;
  if (!state.nearest)
    state.refusal = "MXCSR rounding is not nearest-even";
  else if (!state.subnormal)
    state.refusal = "MXCSR FTZ/DAZ does not preserve subnormals";
  else if (!state.masked)
    state.refusal = "MXCSR exception classes are not all masked";
  else if (!state.precision)
    state.refusal = "MSVC x64 declared long double evaluation is unsupported";
#elif defined(__APPLE__) && defined(__aarch64__)
  state.adapter = "apple-arm64-fpcr";
  state.supported = std::numeric_limits<long double>::digits == 53;
  state.has_arm = true;
  asm volatile("mrs %0, fpcr" : "=r"(state.fpcr) : : "memory");
  asm volatile("mrs %0, fpsr" : "=r"(state.fpsr) : : "memory");
  state.nearest = (state.fpcr & 0x00c00000u) == 0;
  // FZ/FIZ cover binary32/64; FZ16 also covers accepted half-precision input
  // conversion. AH selects alternate evaluation behavior and is not admitted
  // by this adapter. Unimplemented optional bits read as zero.
  state.subnormal = (state.fpcr & 0x01080001u) == 0;
  state.masked = (state.fpcr & 0x00009f00u) == 0;
  state.precision = state.supported && (state.fpcr & 0x2u) == 0;
  if (!state.nearest)
    state.refusal = "FPCR rounding is not nearest-even";
  else if (!state.subnormal)
    state.refusal = "FPCR flush controls do not preserve subnormals";
  else if (!state.masked)
    state.refusal = "FPCR exception traps are enabled";
  else if (!state.precision)
    state.refusal = "FPCR or declared long double evaluation is unsupported";
#endif
  if (state.compatible()) state.refusal = "";
  return state;
}

class EnvironmentError : public std::runtime_error {
 public:
  explicit EnvironmentError(const State& state)
      : std::runtime_error(std::string("native runtime FP profile: ") +
                           state.refusal) {}
};

PYVORO2_RUNTIME_NOINLINE inline void require_environment() {
  const State state = inspect();
  if (!state.compatible()) throw EnvironmentError(state);
}

inline py::dict state_dictionary() {
  const State state = inspect();
  py::dict result;
  result["adapter"] = state.adapter;
  result["supported"] = state.supported;
  result["compatible"] = state.compatible();
  result["nearest"] = state.nearest;
  result["subnormal"] = state.subnormal;
  result["masked"] = state.masked;
  result["precision"] = state.precision;
  result["reason"] = state.refusal;
  result["x87_control"] = state.has_x87 ? py::cast(state.x87_control) : py::none();
  result["x87_status"] = state.has_x87 ? py::cast(state.x87_status) : py::none();
  result["mxcsr"] = state.has_mxcsr ? py::cast(state.mxcsr) : py::none();
  result["fpcr"] = state.has_arm ? py::cast(state.fpcr) : py::none();
  result["fpsr"] = state.has_arm ? py::cast(state.fpsr) : py::none();
  return result;
}

inline void require_no_arguments(const py::args& arguments,
                                 const py::kwargs& keywords) {
  if (PyTuple_GET_SIZE(arguments.ptr()) != 0 || PyDict_Size(keywords.ptr()) != 0)
    throw py::type_error("native inspection takes no arguments");
}

template<class F>
void inspection_def(py::module_& module, const char* name, F function) {
  // Even a no-argument pybind signature formats unexpected values on failure.
  // Raw args/kwargs keep malformed metadata calls callback-free under hostile
  // controls, while valid inspection remains available in that environment.
  module.def(name, [function = std::move(function)](py::args arguments,
                                                   py::kwargs keywords) {
    require_no_arguments(arguments, keywords);
    return function();
  });
}

inline void bind_inspection(py::module_& module) {
  // These methods must not use guarded_def: reporting is safe under hostile
  // controls, and the Python coercion helper calls the no-argument guard.
  inspection_def(module, "_runtime_fp_state", &state_dictionary);
  module.def("_require_runtime_environment", [](py::args arguments,
                                                py::kwargs keywords) {
    try { require_environment(); }
    catch (const EnvironmentError& error) { throw py::value_error(error.what()); }
    require_no_arguments(arguments, keywords);
  });
  inspection_def(module, "_qualification_identity", [] {
    py::dict result;
    result["record_schema"] = "pyvoro2-native-qualification-v1";
    result["policy_revision"] = "issue88-p1";
#ifdef PYVORO2_QUALIFICATION_SOURCE_SHA256
    result["source_sha256"] = PYVORO2_QUALIFICATION_SOURCE_SHA256;
#else
    result["source_sha256"] = "";
#endif
#ifdef PYVORO2_QUALIFICATION_SCHEMA_SHA256
    result["schema_sha256"] = PYVORO2_QUALIFICATION_SCHEMA_SHA256;
#else
    result["schema_sha256"] = "";
#endif
#ifdef PYVORO2_QUALIFICATION_CONSUMER_SHA256
    result["consumer_sha256"] = PYVORO2_QUALIFICATION_CONSUMER_SHA256;
#else
    result["consumer_sha256"] = "";
#endif
    return result;
  });
}

inline void register_imported_module(py::module_& module) {
  // pybind11 >= 3 executes module initialization after import machinery sets
  // __spec__, __file__, and sys.modules. Freeze the loaded file's lifetime here,
  // before returning it to callers or admitting any foreign numeric callbacks.
  require_environment();
  try {
    auto qualification = py::module_::import("pyvoro2._internal.native_qualification");
    require_environment();
    py::object register_native = qualification.attr("register_native");
    require_environment();
    register_native(module);
    require_environment();
  } catch (const py::error_already_set&) {
    // Registration/import callbacks can also change controls and then raise.
    require_environment();
    throw;
  }
}

enum class Family { safety, spatial, planar, ghost, locate };

[[noreturn]] inline void entry_failure(Family family, const EnvironmentError& error) {
  const std::string detail = error.what();
  if (family == Family::ghost)
    throw py::value_error("ghost_native:native:None:GHOST_NATIVE_UNSUPPORTED:" + detail);
  if (family == Family::locate)
    throw py::value_error("locate_native:profile:None:LOCATE_NATIVE_UNSUPPORTED:" + detail);
  if (family == Family::planar) {
    const char* reason = detail.find("rounding") != std::string::npos ? "rounding" :
                         detail.find("subnormal") != std::string::npos ? "subnormal" :
                         "runtime";
    throw std::runtime_error(std::string("planar_certification:profile:") +
                             reason + ": " + detail);
  }
  if (family == Family::spatial) throw std::runtime_error(detail);
  throw py::value_error(detail);
}

inline Family entry_family(py::module_& module, const char* name) {
  const std::string entry(name);
  if (entry.find("ghost") != std::string::npos) return Family::ghost;
  if (entry.find("locate") != std::string::npos) return Family::locate;
  if (entry.find("_observe_") == 0) return Family::spatial;
  const std::string module_name = py::str(module.attr("__name__"));
  if (module_name.find("_core2d") != std::string::npos) return Family::planar;
  return Family::safety;
}

template<class T> using Value = std::remove_cv_t<std::remove_reference_t<T>>;
template<class T> struct Scalar { using type = Value<T>; };
template<class T, std::size_t N> struct Scalar<std::array<T, N>> : Scalar<T> {};
template<class T, class A> struct Scalar<std::vector<T, A>> : Scalar<T> {};
template<class T, class... Rest> struct Scalar<std::tuple<T, Rest...>> : Scalar<T> {};
template<class T> struct IsSequence : std::false_type {};
template<class T, std::size_t N> struct IsSequence<std::array<T, N>> : std::true_type {};
template<class T, class A> struct IsSequence<std::vector<T, A>> : std::true_type {};
template<class... T> struct IsSequence<std::tuple<T...>> : std::true_type {};
template<class T> struct ArrayKind { static constexpr int value = 0; };
template<class T, int Flags> struct ArrayKind<py::array_t<T, Flags>> {
  static constexpr int value = std::is_same_v<T, bool> ? 3 :
                               std::is_floating_point_v<T> ? 1 : 2;
};

template<class T> constexpr const char* argument_kind() {
  using V = Value<T>;
  using S = typename Scalar<V>::type;
  if constexpr (ArrayKind<V>::value == 1) return "float_array";
  else if constexpr (ArrayKind<V>::value == 2) return "int_array";
  else if constexpr (ArrayKind<V>::value == 3) return "bool_array";
  else if constexpr (IsSequence<V>::value)
    return std::is_same_v<S, bool> ? "bool_sequence" :
           std::is_floating_point_v<S> ? "float_sequence" : "int_sequence";
  else if constexpr (std::is_same_v<V, bool>) return "bool";
  else if constexpr (std::is_floating_point_v<V>) return "float";
  else if constexpr (std::is_integral_v<V>) return "int";
  else return "object";
}

inline py::object argument_preparer() {
  require_environment();
  auto module = py::module_::import("pyvoro2._internal.native_runtime");
  require_environment();
  py::object prepare = module.attr("prepare_native_argument");
  require_environment();
  return prepare;
}

template<class T> Value<T> guarded_cast(py::handle source, py::handle prepare) {
  require_environment();
  // Python materializes each foreign protocol/scalar result and checks the
  // controls before NumPy or another caster may continue numeric conversion.
  py::object canonical = prepare(source, argument_kind<T>());
  require_environment();
  auto value = py::cast<Value<T>>(canonical);
  require_environment();
  return value;
}

template<class T> Value<T> guarded_cast(py::handle source) {
  return guarded_cast<T>(source, argument_preparer());
}

template<class F> struct Signature : Signature<decltype(&F::operator())> {};
template<class C, class R, class... A>
struct Signature<R(C::*)(A...) const> { using type = R(A...); };
template<class R, class... A>
struct Signature<R(*)(A...)> { using type = R(A...); };

struct Argument {
  std::string name;
  py::object default_value;
};

inline Argument argument(const py::arg& value) {
  return {value.name == nullptr ? "" : value.name, py::object()};
}

inline Argument argument(const py::arg_v& value) {
  return {value.name == nullptr ? "" : value.name, value.value};
}

template<std::size_t N>
std::array<py::object, N> bind_arguments(
    const std::array<Argument, N>& specification, const py::args& positional,
    const py::kwargs& keywords) {
  // The caller has already inspected the raw environment. Do not ask pybind
  // to format an arity error: repr(float/array) can trap, and a foreign repr
  // can change controls before formatting continues. These constant errors
  // also avoid all repr/str/hash/equality callbacks on the supplied objects.
  std::array<py::object, N> bound;
  const auto count = static_cast<std::size_t>(PyTuple_GET_SIZE(positional.ptr()));
  if (count > N) throw py::type_error("native call has too many positional arguments");
  for (std::size_t i = 0; i < count; ++i)
    bound[i] = py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(positional.ptr(), i));

  Py_ssize_t position = 0;
  PyObject* key;
  PyObject* value;
  while (PyDict_Next(keywords.ptr(), &position, &key, &value)) {
    if (!PyUnicode_Check(key)) throw py::type_error("native keyword name must be a string");
    std::size_t match = N;
    for (std::size_t i = 0; i < N; ++i) {
      // This Unicode C API compares the string's stored code points directly;
      // it never calls a str subclass's __eq__, __hash__, __str__ or __repr__.
      if (!specification[i].name.empty() &&
          PyUnicode_CompareWithASCIIString(key, specification[i].name.c_str()) == 0) {
        match = i;
        break;
      }
    }
    if (match == N) throw py::type_error("native call has an unexpected keyword argument");
    if (bound[match]) throw py::type_error("native call has duplicate argument values");
    bound[match] = py::reinterpret_borrow<py::object>(value);
  }
  for (std::size_t i = 0; i < N; ++i) {
    if (!bound[i]) {
      if (!specification[i].default_value)
        throw py::type_error("native call is missing a required argument");
      bound[i] = specification[i].default_value;
    }
  }
  return bound;
}

template<class> struct Dispatch;
template<class R, class... A> struct Dispatch<R(A...)> {
  static constexpr std::size_t argument_count = sizeof...(A);

  template<std::size_t... I>
  static auto convert(const std::array<py::object, sizeof...(A)>& arguments,
                      py::handle prepare, std::index_sequence<I...>) {
    // Braced initializer evaluation is sequenced left-to-right. Only
    // canonical callback-free objects reach pybind's numeric casters.
    return std::tuple<Value<A>...>{guarded_cast<A>(arguments[I], prepare)...};
  }

  template<class F> static auto wrap(F function, Family family,
                                     py::object module, std::string name,
                                     std::array<Argument, sizeof...(A)> specification) {
    return [function = std::move(function), family, module = std::move(module),
            name = std::move(name), specification = std::move(specification)]
           (py::args positional, py::kwargs keywords) -> R {
      try {
        require_environment();
        auto arguments = bind_arguments(specification, positional, keywords);
        py::object prepare = argument_preparer();
        auto converted = convert(arguments, prepare, std::index_sequence_for<A...>{});
        require_environment();
        auto runtime = py::module_::import("pyvoro2._internal.native_runtime");
        require_environment();
        py::object admit = runtime.attr("admit_native_entry");
        require_environment();
        // The actual owning module, including an already retained older
        // module object, is bound here rather than rediscovered via sys.modules.
        // Canonical shape/selector values distinguish zero-work and ID-only
        // routes without invoking another foreign conversion.
        admit(module, name, py::cast(converted));
        require_environment();
        return std::apply(function, std::move(converted));
      } catch (const EnvironmentError& error) {
        entry_failure(family, error);
      } catch (const py::error_already_set&) {
        // A callback may change controls and then raise. Do not continue
        // through Python numeric error formatting under the changed state.
        const State state = inspect();
        if (!state.compatible()) entry_failure(family, EnvironmentError(state));
        throw;
      }
    };
  }
};

template<class F, class... Extra>
void guarded_def(py::module_& module, const char* name, F function, Extra&&... extra) {
  static_assert(((std::is_same_v<Value<Extra>, py::arg> ||
                  std::is_same_v<Value<Extra>, py::arg_v>) && ...),
                "guarded native annotations must be argument names/defaults");
  static_assert(sizeof...(Extra) ==
                    Dispatch<typename Signature<F>::type>::argument_count,
                "guarded native argument metadata must cover the full signature");
  module.def(name, Dispatch<typename Signature<F>::type>::wrap(
                      std::move(function), entry_family(module, name), module, name,
                      {argument(extra)...}));
}

}  // namespace pyvoro2::native_runtime
#undef PYVORO2_RUNTIME_NOINLINE
#endif
