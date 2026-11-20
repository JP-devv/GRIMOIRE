#include "qfn_verify_split_plan.hpp"

using b70::qfn_verify_split_count;

static_assert(qfn_verify_split_count(true, true, false, true, true, 4, 2706, 32768, 6, 1, 256) == 32);
static_assert(qfn_verify_split_count(false, true, false, true, true, 4, 2706, 32768, 6, 1, 256) == 0);
static_assert(qfn_verify_split_count(true, false, false, true, true, 4, 2706, 32768, 6, 1, 256) == 0);
static_assert(qfn_verify_split_count(true, true, true, true, true, 4, 2706, 32768, 6, 1, 256) == 0);
static_assert(qfn_verify_split_count(true, true, false, false, true, 4, 2706, 32768, 6, 1, 256) == 0);
static_assert(qfn_verify_split_count(true, true, false, true, false, 4, 2706, 32768, 6, 1, 256) == 0);
static_assert(qfn_verify_split_count(true, true, false, true, true, 4, 2047, 32768, 6, 1, 256) == 0);
static_assert(qfn_verify_split_count(true, true, false, true, true, 4, 2048, 32768, 6, 1, 256) == 32);
static_assert(qfn_verify_split_count(true, true, false, true, true, 4, 32764, 32768, 6, 1, 256) == 32);
static_assert(qfn_verify_split_count(true, true, false, true, true, 4, 32765, 32768, 6, 1, 256) == 0);
static_assert(qfn_verify_split_count(true, true, false, true, true, 3, 2706, 32768, 6, 1, 256) == 0);
static_assert(qfn_verify_split_count(true, true, false, true, true, 4, 2706, 32768, 24, 2, 256) == 0);
static_assert(qfn_verify_split_count(true, true, false, true, true, 4, 2706, 32768, 6, 1, 128) == 0);

int main() {}
