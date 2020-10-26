#pragma once

// Pybind11 headers must be included consistently,
// otherwise you get a One-definition rule violation.
// https://github.com/pybind/pybind11/issues/1055
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utils/signal.h>

#include <boost/array.hpp>

namespace py = pybind11;

/*
// Specialization of pybind type_caster for boost::array.
template <typename Type, size_t Size>
struct py::detail::type_caster<boost::array<Type, Size>>
    : py::detail::array_caster<boost::array<Type, Size>, Type, false, Size> { };
*/


namespace pybind11 {
namespace detail {
template <typename Type, size_t Size>
struct type_caster<boost::array<Type, Size>>
    : array_caster<boost::array<Type, Size>, Type, false, Size> { };
}  // namespace detail
}  // namespace pybind11

namespace epidemics {

class SignalRAII {
public:
    SignalRAII() {
        check_signals_func = []() {
            // https://stackoverflow.com/questions/14707049/allowing-ctrl-c-to-interrupt-a-python-c-extension
            if (PyErr_CheckSignals() != 0)
                throw std::runtime_error("Signal received. Breaking.");
        };
    }
    ~SignalRAII() {
        check_signals_func = nullptr;
    }
};

}  // namespace epidemics
