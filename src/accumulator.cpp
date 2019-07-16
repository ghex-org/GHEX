

#include "accumulator.hpp"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>

namespace gridtools {
namespace detail {
struct accumulator_impl
{
    boost::accumulators::accumulator_set<
        double, 
        boost::accumulators::stats<
            boost::accumulators::tag::mean, 
            boost::accumulators::tag::variance(boost::accumulators::lazy), 
            boost::accumulators::tag::max, 
            boost::accumulators::tag::min> > m_acc;
};
}}

gridtools::accumulator::accumulator()
: m_impl(new detail::accumulator_impl())
{}
gridtools::accumulator::~accumulator()
{
    if (!(this->m_moved))
        delete this->m_impl;
}
gridtools::accumulator::accumulator(const accumulator& other)
: m_impl( new detail::accumulator_impl(*other.m_impl) )
{}
gridtools::accumulator::accumulator(accumulator&& other)
: m_impl( other.m_impl )
{
    other.m_moved = true;
}

void gridtools::accumulator::operator()(double x)
{
    this->m_impl->m_acc(x);
}
double gridtools::accumulator::mean()
{
    return boost::accumulators::mean(this->m_impl->m_acc);
}
double gridtools::accumulator::stdev()
{
    return std::sqrt(boost::accumulators::variance(this->m_impl->m_acc));
}
double gridtools::accumulator::min()
{
    return boost::accumulators::min(this->m_impl->m_acc);
}
double gridtools::accumulator::max()
{
    return boost::accumulators::max(this->m_impl->m_acc);
}
