#pragma once

#include <numeric>
#include <optional>

#include <Eigen/Core>


namespace softee {
    
using Regression = Eigen::Vector<double, 1>;

class RegressionData
{
public:

    RegressionData(
        double alpha,
        const std::optional<Regression>& lower,
        const std::optional<Regression>& upper,
        const Targets& targets)
        : G_{0, 0}
        , H_{{0, 0}, {0, 0}}
        , G_reg_{0, 0}
        , H_reg_{{alpha, -alpha}, {-alpha, alpha}}
    {
        if (lower) {
            const auto& lower_value = *lower;
            G_reg_(0) = -alpha * lower_value(0);
            H_reg_(0, 0) += alpha;
        }
        if (upper) {
            const auto& upper_value = *upper;
            G_reg_(1) = -alpha * upper_value(0);
            H_reg_(1, 1) += alpha;
        }

        G_(1) = -std::reduce(begin(targets), end(targets));
        H_(1, 1) = targets.size();
    }

    void pass_sample(double y)
    {
        G_(0) -= y;
        G_(1) += y;

        H_(0, 0) += 1.0;
        H_(1, 1) -= 1.0;
    }

    Eigen::Vector<double, 2>    G_tot() const { return G_ + G_reg_; }

    Eigen::Matrix<double, 2, 2> H_tot() const { return H_ + H_reg_; }

private:

    Eigen::Vector<double, 2>    G_;
    Eigen::Matrix<double, 2, 2> H_;
    Eigen::Vector<double, 2>    G_reg_;
    Eigen::Matrix<double, 2, 2> H_reg_;
};



struct RegressionTraits
{
    using Leaf = Regression;
    using Data = RegressionData;

    static Leaf calculate_initial(const Targets& targets)
    {
        Leaf w{std::reduce(begin(targets), end(targets)) / targets.size()};
        return w;
    }
};

} // namespace softee
