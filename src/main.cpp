
#include "post_processing.hpp"
#include "pre_processing.hpp"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct image
{
    std::vector<uint8_t> data;
    int width;
    int height;
    int channels;
};

image load_image(const std::filesystem::path& image_path)
{
    int width, height, channels;
    uint8_t* image_data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if (!image_data)
    {
        throw std::runtime_error("Failed to load image: " + image_path.string());
    }

    size_t total_pixels = width * height * channels;
    image img{
        .data = std::vector<uint8_t>(total_pixels),
        .width = width,
        .height = height,
        .channels = channels};

    std::memcpy(img.data.data(), image_data, total_pixels);
    stbi_image_free(image_data);
    return img;
}

std::string print_shape(const std::vector<std::int64_t>& v)
{
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

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

int main()
{
    const std::basic_string<ORTCHAR_T> model_path = "squeezenet1.1-7.onnx";
    // const std::basic_string<ORTCHAR_T> model_path = "resnet18-v1-7.onnx";
    const std::filesystem::path image_path = "images/dog.png";

    std::cout
        << "\nModel: " << model_path
        << "\nImage: " << image_path
        << std::endl;

    // ResNet & SqueezeNet target size
    const int target_width = 224;
    const int target_height = 224;

    image img = load_image(image_path);
    std::vector<float> img_blob = preprocess<float>(
        img.data,
        img.width,
        img.height,
        img.channels,
        target_width,
        target_height,
        1.0 / 255.0,
        std::vector<float>{0.485f, 0.456f, 0.406f},
        std::vector<float>{0.229f, 0.224f, 0.225f},
        true);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ResNet");
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, model_path.c_str(), session_options);

    // Inputs
    std::vector<const char*> input_names;
    std::vector<std::string> input_names_str;
    std::vector<std::vector<std::int64_t>> input_shapes;
    for (std::size_t i = 0; i < session.GetInputCount(); i++)
    {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        auto input_shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        auto input_type = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
        auto input_count = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementCount();

        for (auto& s : input_shape)
        {
            if (s < 0)
                s = 1;
        }
        input_shapes.emplace_back(input_shape);
        input_names_str.emplace_back(input_name.get());

        std::cout
            << "\nInput: " << i
            << "\n - name: " << input_name
            << "\n - shape: " << print_shape(input_shape)
            << "\n - element type: " << input_type
            << "\n - element count: " << input_count
            << std::endl;
    }

    for (auto&& s : input_names_str)
    {
        input_names.emplace_back(s.c_str());
    }

    // Outputs
    std::vector<const char*> output_names;
    std::vector<std::string> output_names_str;
    std::vector<std::vector<std::int64_t>> output_shapes;
    for (std::size_t i = 0; i < session.GetOutputCount(); i++)
    {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        auto output_shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        auto output_type = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
        auto output_count = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementCount();

        for (auto& s : output_shape)
        {
            if (s < 0)
                s = 1;
        }
        output_shapes.emplace_back(output_shape);
        output_names_str.emplace_back(output_name.get());

        std::cout
            << "\nOutput: " << i
            << "\n - name: " << output_name
            << "\n - shape: " << print_shape(output_shape)
            << "\n - element type: " << output_type
            << "\n - element count: " << output_count
            << std::endl;
    }

    for (auto&& s : output_names_str)
    {
        output_names.emplace_back(s.c_str());
    }

    // Create inference tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(
        Ort::Value::CreateTensor<float>(
            memory_info,
            img_blob.data(),
            img_blob.size(),
            input_shapes.at(0).data(),
            input_shapes.at(0).size()));

    // Run inference
    std::vector<Ort::Value> output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_tensors.data(),
        input_names.size(),
        output_names.data(),
        output_names.size());

    // Postprocess output
    float* outputs_raw = output_tensors.front().GetTensorMutableData<float>();
    std::vector<int64_t> output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    // Print results
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

    std::cout << "\nPredicted Label ID: " << pred_id << std::endl;
    std::cout << "Predicted Label: " << pred_label << std::endl;
    std::cout << "Uncalibrated Confidence: " << confidence << std::endl;

    return EXIT_SUCCESS;
}
