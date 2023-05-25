
#include <iostream>
#include <fstream>

#include <spdlog/spdlog.h>

#include "softee/alpha_generators.hpp"
#include "softee/forward.hpp"
#include "softee/regression.hpp"
#include "softee/tree_io.hpp"
#include "softee/types.hpp"


namespace softee {

std::tuple<Features, Targets, FeatureNames> load_data(std::string file_loc)
{
    auto handle = std::fstream(file_loc, std::ios_base::in);

    std::string header;
    std::getline(handle, header);

    FeatureNames feature_names;
    std::size_t target_loc;
   
    {
        std::stringstream ss{header};
        std::string header_item;
        std::size_t loc = 0;
        while (std::getline(ss, header_item, ',')) {
            if (header_item == "Survived") {
                target_loc = loc;
            }
            else {
                feature_names[loc] = header_item;
            }

            loc++;
        }
    }

    Features features;
    Targets targets;

    std::string line;
    while (std::getline(handle, line))
    {
        std::stringstream ss{line};
        std::string field;
        for (std::size_t feature_id = 0; std::getline(ss, field, ','); ++feature_id) {
            if (feature_id == target_loc) {
                targets.push_back(std::stod(field));
            }
            else {
                features[feature_id].push_back(std::stod(field));
            }
        }
    }

    return {features, targets, feature_names};
}

} // namespace softee

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cout << "Usage: ./titanic {data-file}" << std::endl;
        return 1;
    }

    using namespace softee;
    auto [features, targets, feature_names] = load_data(argv[1]);

    Config config{5, 10};
    ConstantAlphaGenerator alpha_generator{0.25};

    ForwardAlgorithm<RegressionTraits> builder{config, alpha_generator};

    //spdlog::set_level(spdlog::level::debug)
    auto tree = builder.build_tree( features, targets);
    print(std::cout, tree, feature_names);
    return 0;
}
