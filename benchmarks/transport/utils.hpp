#ifndef INCLUDED_GHEX_UTILS_HPP
#define INCLUDED_GHEX_UTILS_HPP

template<typename Msg>
void make_zero(Msg& msg) {
    for (auto& c : msg)
	c = 0;
}

#endif /* INCLUDED_GHEX_UTILS_HPP */

