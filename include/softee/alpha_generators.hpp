#pragma once

#include "softee/types.hpp"

namespace softee {

class ConstantAlphaGenerator : public IAlphaGenerator
{
public:
    explicit ConstantAlphaGenerator(double alpha)
        : alpha_{alpha}
    {
    }

    double operator () (std::size_t) const override { return alpha_; }

private:

    double alpha_;
};


class LinearAlphaGenerator : public IAlphaGenerator
{
public:
     LinearAlphaGenerator(double alpha, double increment)
        : alpha_{alpha}
        , increment_{increment}
    {
    }

    double operator () (std::size_t depth) const override
    {
        return alpha_ + static_cast<double>(depth) * increment_;
    }

private:

    double alpha_;
    double increment_;
};

} // namespace softee
