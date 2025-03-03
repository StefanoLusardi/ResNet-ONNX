#pragma once

#include <array>
#include <cstdint>
#include <vector>

std::vector<uint8_t> resize_image(
    const std::vector<std::uint8_t>& image,
    int image_width,
    int image_height,
    int image_channels,
    int target_width,
    int target_height)
{
    std::vector<uint8_t> resized_image(target_width * target_height * image_channels);

    double scale_x = static_cast<double>(image_width) / target_width;
    double scale_y = static_cast<double>(image_height) / target_height;

    for (int y = 0; y < target_height; ++y)
    {
        for (int x = 0; x < target_width; ++x)
        {
            int srcX = std::min(static_cast<int>(x * scale_x), image_width - 1);
            int srcY = std::min(static_cast<int>(y * scale_y), image_height - 1);

            for (int c = 0; c < image_channels; ++c)
            {
                resized_image[(y * target_width + x) * image_channels + c] =
                    image[(srcY * image_width + srcX) * image_channels + c];
            }
        }
    }

    return resized_image;
}

void resize_image_aspect_ratio(
    const std::vector<std::uint8_t>& image,
    int image_width,
    int image_height,
    int image_channels,
    int target_width,
    int target_height,
    std::vector<std::uint8_t>& resized_image)
{
    // Calculate the aspect ratios
    double aspect_ratio_image = static_cast<double>(image_width) / image_height;
    double aspect_ratio_target = static_cast<double>(target_width) / target_height;

    // Determine the scaling factors and new dimensions
    int new_width;
    int new_height;
    if (aspect_ratio_image > aspect_ratio_target)
    {
        new_width = target_width;
        new_height = static_cast<int>(target_width / aspect_ratio_image);
    }
    else
    {
        new_height = target_height;
        new_width = static_cast<int>(target_height * aspect_ratio_image);
    }

    // Calculate padding
    int pad_x = (target_width - new_width) / 2;
    int pad_y = (target_height - new_height) / 2;

    // Scale factors
    double scale_x = static_cast<double>(image_width) / new_width;
    double scale_y = static_cast<double>(image_height) / new_height;

    // Resize with aspect ratio preservation
    for (int y = 0; y < new_height; ++y)
    {
        for (int x = 0; x < new_width; ++x)
        {
            int src_x = std::min(static_cast<int>(x * scale_x), image_width - 1);
            int src_y = std::min(static_cast<int>(y * scale_y), image_height - 1);

            for (int c = 0; c < image_channels; ++c)
            {
                auto idx = ((y + pad_y) * target_width + (x + pad_x)) * image_channels + c;
                resized_image[idx] = image[(src_y * image_width + src_x) * image_channels + c];
            }
        }
    }
}

std::vector<std::uint8_t> resize_image_aspect_ratio(
    const std::vector<std::uint8_t>& image,
    int image_width,
    int image_height,
    int image_channels,
    int target_width,
    int target_height)
{
    std::vector<std::uint8_t> resized_image(target_width * target_height * image_channels, std::uint8_t(0));
    resize_image_aspect_ratio(image, image_width, image_height, image_channels, target_width, target_height, resized_image);
    return resized_image;
}

template <typename T>
void create_blob(
    const std::vector<std::uint8_t>& image,
    int image_width,
    int image_height,
    int image_channels,
    std::vector<T>& blob,
    T normalize_factor = 1.0 / 255.0,
    const std::vector<T>& mean = {0.0, 0.0, 0.0},
    const std::vector<T>& scale = {1.0f, 1.0f, 1.0f},
    bool swapRB_channels = false)
{
    (void)scale;
    for (int c = 0; c < image_channels; ++c)
    {
        const int channel_offset = (swapRB_channels ? (2 - c) : c);

        for (int y = 0; y < image_height; ++y)
        {
            for (int x = 0; x < image_width; ++x)
            {
                const int idx_offset = y * image_width + x;
                const int blob_idx = c * image_height * image_width + idx_offset;
                const int image_idx = idx_offset * image_channels + channel_offset;
                blob[blob_idx] = (static_cast<T>(image[image_idx]) * normalize_factor - mean[c]) / scale[c];
            }
        }
    }
}

template <typename T>
std::vector<T> create_blob(
    const std::vector<std::uint8_t>& image,
    int image_width,
    int image_height,
    int image_channels,
    T normalize_factor = 1.0 / 255.0,
    const std::vector<T>& mean = {0.0, 0.0, 0.0},
    const std::vector<T>& scale = {1.0f, 1.0f, 1.0f},
    bool swapRB_channels = false)
{
    std::vector<float> blob(image_channels * image_width * image_height);
    create_blob(image, image_width, image_height, image_channels, blob, normalize_factor, mean, scale, swapRB_channels);
    return blob;
}

template <typename T>
std::vector<T> preprocess(
    const std::vector<std::uint8_t>& image,
    int image_width,
    int image_height,
    int image_channels,
    int target_width,
    int target_height,
    T normalize_factor = 1.0 / 255.0,
    const std::vector<T>& mean = {0.0f, 0.0f, 0.0f},
    const std::vector<T>& scale = {1.0f, 1.0f, 1.0f},
    bool swapRB_channels = false)
{
    const std::vector<std::uint8_t> resized_image = resize_image(image, image_width, image_height, image_channels, target_width, target_height);
    const std::vector<T> blob = create_blob(resized_image, target_width, target_height, image_channels, normalize_factor, mean, scale, swapRB_channels);
    return blob;
}
