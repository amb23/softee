#pragma once

#include <numeric>
#include <optional>
#include <unordered_map>
#include <vector>

#include <Eigen/Cholesky>
#include <spdlog/spdlog.h>

#include "softee/trees.hpp"
#include "softee/types.hpp"


namespace softee {

template <typename Leaf>
struct Bound
{
    double lower_bound = std::numeric_limits<double>::min();
    std::optional<Leaf> lower_value = std::nullopt;

    double upper_bound = std::numeric_limits<double>::max();
    std::optional<Leaf> upper_value = std::nullopt;
};


template <typename Leaf>
class Bounds
{
public:
    const Bound<Leaf>* get(unsigned int feature_id) const
    {
        if (auto it = data_.find(feature_id); it != data_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    Bound<Leaf>& get_or_create(unsigned int feature_id)
    {
        return data_[feature_id];
    }

    void add_upper_bound(unsigned int feature_id, double x, Leaf w)
    {
        auto& bound = data_[feature_id];
        if (bound.upper_bound > x) {
            bound.upper_bound = x;
            bound.upper_value = std::move(w);
        }
    }

    void add_lower_bound(unsigned int feature_id, double x, Leaf w)
    {
        auto& bound = data_[feature_id];
        if (bound.lower_bound < x) {
            bound.lower_bound = x;
            bound.lower_value = std::move(w);
        }
    }

private:

    std::unordered_map<unsigned int, Bound<Leaf>> data_;

};


template <typename Traits>
class ForwardAlgorithm
{
public:

    using Leaf = Traits::Leaf;
    using Tree = FlatTree<Leaf>;

    ForwardAlgorithm(
        const Config& config,
        const IAlphaGenerator& alpha_generator)
        : config_{config}
        , alpha_generator_{alpha_generator}
    {
    }

    Tree build_tree(
        const Features& features,
        const Targets& targets) const
    {
        Tree tree{config_.max_depth};
        auto initial = Traits::calculate_initial(targets);
        double cost = std::numeric_limits<double>::max(); // FIXME
        fill(0u, tree, initial, cost, features, targets, Bounds<Leaf>{});

        return tree;
    }


private:

    struct FitResult
    {
        unsigned int feature_id;

        // The split occurs over [x_lower, x_upper]
        double x_lower;
        double x_upper;

        Leaf w_lower;
        Leaf w_upper;
    };


    using PreparedFeature = std::vector<std::pair<double, double>>;

    PreparedFeature prepare_feature(
        const Feature& feature,
        const Targets& targets) const
    {
        PreparedFeature prepared_feature;
        prepared_feature.reserve(feature.size());

        for (std::size_t i = 0; i < feature.size(); ++i) {
            prepared_feature.emplace_back(feature[i], targets[i]);
        }

        std::sort(begin(prepared_feature), end(prepared_feature));
        return prepared_feature;
    }

    void fill(
        std::size_t loc,
        Tree& tree,
        const Leaf& current,
        double current_cost,
        const Features& features,
        const Targets& targets,
        const Bounds<Leaf>& bounds) const
    {
        if (tree.get_depth(loc) >= config_.max_depth ||
            targets.size() < config_.min_samples) {
            SPDLOG_DEBUG("Completed fit at loc {}", loc);
            tree[loc] = current;
            return;
        }

        auto [fit_result, cost] = best_fit(
            alpha_generator_(tree.get_depth(loc)),
            current,
            features,
            targets,
            bounds
        );

        if (current_cost < cost) {
            SPDLOG_DEBUG("Cannot improve on current cost at loc {}", loc);
            tree[loc] = current;
            return;
        }

        tree[loc] = Split{
            fit_result.feature_id,
            fit_result.x_lower,
            fit_result.x_upper
        };

        fill_next(true,  loc, tree, fit_result, features, targets, bounds);
        fill_next(false, loc, tree, fit_result, features, targets, bounds);
    }

    std::pair<Features, Targets> split_data(
        bool is_below,
        const FitResult& split,
        const Features& features,
        const Targets& targets) const
    {
        std::vector<std::size_t> index;
        const auto& splitting_feature = features.find(split.feature_id)->second;

        for (std::size_t i = 0; i < splitting_feature.size(); ++i) {
            double x = splitting_feature[i];
            if (is_below == x < split.x_upper) {
                index.push_back(i);
            }
        }

        Features split_features_;
        for (const auto& [feature_id, feature] : features) {
            Feature split_feature;
            split_feature.reserve(index.size());
            for (std::size_t i : index) {
                split_feature.push_back(feature[i]);
            }
            split_features_.emplace(feature_id, std::move(split_feature));
        }

        Targets split_targets_;
        split_targets_.reserve(index.size());
        for (std::size_t i : index) {
            split_targets_.push_back(targets[i]);
        }

        return {split_features_, split_targets_};
    }

    void fill_next(
        bool is_lower,
        std::size_t loc,
        Tree& tree,
        const FitResult& fit_result,
        const Features& features,
        const Targets& targets,
        const Bounds<Leaf>& current_bounds) const
    {
        auto [next_features, next_targets] = split_data(
            is_lower,
            fit_result,
            features,
            targets
        );

        Bounds<Leaf> bounds = current_bounds;
        auto bound = bounds.get_or_create(fit_result.feature_id);
        if (is_lower) {
            bound.upper_bound = fit_result.x_lower;
            bound.upper_value = fit_result.w_upper;
        }
        else {
            bound.lower_bound = fit_result.x_upper;
            bound.lower_value = fit_result.w_lower;
        }

        fill(
            tree.next(is_lower, loc),
            tree,
            is_lower ? fit_result.w_lower : fit_result.w_upper,
            std::numeric_limits<double>::max(), // FIXME
            next_features,
            next_targets,
            bounds
        );
    }

    std::pair<FitResult, double> best_fit(
        double alpha,
        const Leaf& current,
        const Features& features,
        const Targets& targets,
        const Bounds<Leaf>& bounds) const
    {
        double best_cost = std::numeric_limits<double>::max();
        std::optional<FitResult> best_fit_;

        for (const auto& [feature_id, feature] : features) {

            auto [fit_, cost] = fit_feature(
                feature_id,
                current,
                alpha, 
                feature,
                targets,
                bounds.get(feature_id)
            );

            SPDLOG_DEBUG("Feature {} has cost= {} ", feature_id,  cost);

            if (cost < best_cost) {
                best_fit_ = fit_;
                best_cost = cost;
            }
        }

        // assert(best_fit_)
        return {*best_fit_, best_cost};
    }

    std::tuple<FitResult, double> fit_feature(
        unsigned int feature_id,
        const Leaf& current,
        double alpha,
        const Feature& feature,
        const Targets& targets,
        const Bound<Leaf>* bound) const
    {
        typename Traits::Data data(
            alpha,
            bound ? bound->lower_value : std::optional<Leaf>{},
            bound ? bound->upper_value : std::optional<Leaf>{},
            targets
        );

        auto prepared_feature = prepare_feature(feature, targets);

        FitResult fit_result{
            feature_id,
            std::get<0>(prepared_feature[0]),
            std::get<0>(prepared_feature[1]),
            current,
            current
        };

        double best_cost = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < prepared_feature.size() - 1; ++i) {
            auto [x, y] = prepared_feature[i];
            data.pass_sample(y);

            double x_next = std::get<0>(prepared_feature[i + 1]);

            // FIXME - doubles not equal
            if (x != x_next) {
                const auto& G = data.G_tot();
                const auto& H = data.H_tot();
                auto W = solve(G, H);

                double cost_linear = W.transpose() * G;
                double cost_quad = W.transpose() * H * W;
                double cost = cost_linear + 0.5 * cost_quad;
                if (cost < best_cost) {
                    best_cost = cost;

                    std::size_t n = current.rows();
                    fit_result.x_lower = x;
                    fit_result.x_upper = x_next;
                    fit_result.w_lower = W(Eigen::seq(0, n - 1));
                    fit_result.w_upper = W(Eigen::seq(n, 2*n - 1));
                }
            }
        }

        return {fit_result, best_cost};
    }

    template <int N>
    Eigen::Vector<double, N> solve(
        const Eigen::Vector<double, N>& G,
        const Eigen::Matrix<double, N, N>& H) const
    {
        // G + Hx => Hx = -G
        return H.llt().solve(-G);
    }

    Config config_;
    const IAlphaGenerator& alpha_generator_;
};


} // namespace softee
