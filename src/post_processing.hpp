#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

std::vector<std::string> read_labels(const std::string& labels_path)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labels_path);
    while (std::getline(fp, line))
    {
        labels.push_back(line);
    }
    return labels;
}

auto postprocess(const float* outputs_raw) -> std::tuple<int, std::string, float>
{
    int pred_id;
    float activation = 0;
    float maxActivation = std::numeric_limits<float>::lowest();
    float expSum = 0;
    auto labels = read_labels("labels.txt");
    for (int i = 0; i < labels.size(); i++)
    {
        activation = outputs_raw[i];
        expSum += std::exp(activation);
        if (activation > maxActivation)
        {
            pred_id = i;
            maxActivation = activation;
        }
    }

    std::string pred_label = labels.at(pred_id);
    float confidence = std::exp(maxActivation) / expSum;

    return {pred_id, pred_label, confidence};
}
