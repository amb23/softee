#pragma once

#include <unordered_map>
#include <vector>

#include <Eigen/Core>


namespace softee {

using Feature = std::vector<double>;
using Features = std::unordered_map<unsigned int, Feature>;

using FeatureNames = std::unordered_map<unsigned int, std::string>;

using Targets = std::vector<double>;


class IAlphaGenerator
{
public:

    virtual ~IAlphaGenerator() {}

    virtual double operator () (std::size_t depth) const = 0;
};


struct Config
{
    unsigned int max_depth;
    unsigned int min_samples;
};


} // namespace softee
