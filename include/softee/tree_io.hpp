#pragma once

#include <iostream>

#include "softee/trees.hpp"


/// We print to a diagram that can be pasted directly into https://mermaid.live

namespace softee {

inline std::string get_name(unsigned int feature_id, const FeatureNames& names)
{
    if (auto it = names.find(feature_id); it != names.end()) {
        return it->second;
    }
    return "feature_" + std::to_string(feature_id);
}


template <typename Leaf>
void print(
    std::ostream& os,
    const FlatTree<Leaf>& tree,
    std::size_t loc,
    const FeatureNames& feature_names)
{
    if (std::holds_alternative<Leaf>(tree[loc])) {
        os << "  N" << loc << "[ "<< std::get<Leaf>(tree[loc]) << " ]\n"; 
    }
    else {
        const auto& split = std::get<Split>(tree[loc]);
        os << "  N"
            << loc
            << "[ "
            << get_name(split.feature_id, feature_names)
            << " < "
            << 0.5 * (split.x_lower + split.x_upper)
            << " ]\n";
        os << "  N" << loc << " .-> N" << tree.next(true, loc) << "\n";
        os << "  N" << loc << " --> N" << tree.next(false, loc) << "\n";

        print(os, tree, tree.next(true, loc), feature_names);
        print(os, tree, tree.next(false, loc), feature_names);
    }
}


template <typename Leaf>
void print(std::ostream& os, const FlatTree<Leaf>& tree, const FeatureNames& feature_names)
{
    std::cout << "flowchart TD\n";
    print(os, tree, 0, feature_names);
}

template <typename Leaf>
void print(std::ostream& os, const FlatTree<Leaf>& tree)
{
    print(os, tree, FeatureNames{});
}

} // namespace softee
