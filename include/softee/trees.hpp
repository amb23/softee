#pragma once


#include <variant>
#include <vector>


namespace softee {

struct Split
{
    unsigned int feature_id;

    double x_lower;
    double x_upper;
};


template <typename Leaf>
using Node = std::variant<Split, Leaf>;


template <typename Leaf>
class FlatTree
{
public:

    using value_type = Node<Leaf>;
    using container_type = std::vector<value_type>;
    using reference = container_type::reference;
    using iterator = container_type::iterator;
    using size_type = container_type::size_type;


    FlatTree()
        : data_{}
    {
    }

    explicit FlatTree(unsigned long maximum_depth)
        : data_((1 << (maximum_depth + 1)) - 1)
    {
    }

    explicit FlatTree(container_type data)
        : data_(std::move(data))
    {
    }

    unsigned long get_depth() const {
        unsigned long size = data_.size();
        return size > 0 ? get_depth(size - 1) : 0ul;
    }

    unsigned long get_depth(unsigned long loc) const {
        constexpr int max = sizeof(unsigned long) * 8 - 1;
        return max - __builtin_clzl(loc + 1);
    }

    std::size_t next(bool lower, unsigned long loc) const {
        return (loc << 1) + (lower ? 1ul : 2ul);
    }

    decltype(auto) operator[] (size_type loc) { return data_[loc]; }
    decltype(auto) operator[] (size_type loc) const { return data_[loc]; }

    const container_type& data() const { return data_; }

private:

    container_type data_;
};


} // namespace softee
