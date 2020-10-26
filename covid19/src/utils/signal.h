#pragma once

namespace epidemics {

/// For handling Ctrl-C.
using CheckSignalsFunc = void(*)();

// Initially nullptr. If set, it will be called from the solver on every time step.
extern CheckSignalsFunc check_signals_func;

}  // namespace epidemics
