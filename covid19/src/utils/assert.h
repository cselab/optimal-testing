#pragma once

namespace epidemics {

void _die [[gnu::format(printf, 3, 4)]] (const char *filename, int line, const char *fmt, ...);

#define DIE(...) _die(__FILE__, __LINE__, __VA_ARGS__)

}  // namespace epidemics
