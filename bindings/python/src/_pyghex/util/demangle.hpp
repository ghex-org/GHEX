/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <algorithm>
#include <cctype>
#include <cxxabi.h>
#include <string>
#include <memory>
#include <typeinfo>
#include <stdexcept>

namespace pyghex
{
namespace util
{

template<typename T>
std::string
demangle()
{
    int                                    status = 0;
    std::unique_ptr<char, void (*)(void*)> res{
        abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status), std::free};
    if (status != 0) throw std::runtime_error("Could not demangle.");
    return res.get();
}

inline std::string
mangle_python(std::string s) {
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); }), s.end());
    std::string _ghex = "ghex::";
    auto pos = s.find(_ghex);
    while(pos != std::string::npos) {
        s.erase(pos, _ghex.length());
        pos = s.find(_ghex);
    }
    for (auto& c : s) {
        switch(c) {
            case ':':
            case ',':
            case '<':
            case '>':
                c = '_';
                break;
        }
    }
    return s;
}

template<typename T>
std::string
mangle_python()
{
    return mangle_python(demangle<T>());
}

} // namespace util
} // namespace pyghex
