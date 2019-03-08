/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */



#include <prototype/is_multi_arg.hpp>


namespace gt = gridtools;

int main() {
    {
        auto foo = [](int) {};
        static_assert(gt::is_multi_arg<decltype(foo), int>::type::value == false, "");
    }

    {
        auto foo = []() {};
        static_assert(gt::is_multi_arg<decltype(foo), int>::type::value == false, "");
    }

    {
        auto foo = [](int, float, char) {};
        static_assert(gt::is_multi_arg<decltype(foo), int>::type::value == true, "");
    }
}
