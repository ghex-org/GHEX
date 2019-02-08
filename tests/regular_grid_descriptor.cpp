#include <prototype/regular_grid_descriptors.hpp>
#include <iostream>
#include <gridtools/common/layout_map.hpp>
#include <vector>

namespace gt = gridtools;

template <typename ValueType, typename Layout>
class data_t {
public:

    using value_type = ValueType;
    using layout = Layout;

    std::array<int, layout::masked_length> m_sizes;

    template <typename ...Sizes>
    data_t(Sizes... s) : m_sizes{s...} {}

    template <int I>
    int begin() const {
        return 0;
    }

    template <int I>
    int stride() const {
        int stride = 1;
        for (int i = 0; i < I; ++i) {
            stride *= m_sizes[i];
        }
        return stride;
    }

    template <int I>
    int end() const {
        return m_sizes[I];
    }
};

int main() {

    using data_type = data_t<int, gt::layout_map<0,1,2,3,4> >;

    data_type data(2, 3, 4, 5, 6);

    std::array<gt::dimension_descriptor, 2> halos = { gt::dimension_descriptor{ 2, 2, 2, 6}, { 1, 1, 1, 6} };

    gt::regular_grid_descriptor< 2 /* number of dimensions */ > x(halos);

    std::vector<int> container;

    x.pack< gt::partitioned<2,0> >(data,
                                   [&container](data_type::value_type x) {container.push_back(x);},
                                   gt::direction<2>({-1,-1}));

    { // unit test for partitioned
        constexpr gt::partitioned<2,3> p;

        static_assert(p.contains(3));
        static_assert(!p.contains(4));
    }

}
