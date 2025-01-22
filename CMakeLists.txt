cmake_minimum_required(VERSION 3.21)
project(ResNet-ONNX)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build/modules)

find_package(onnxruntime REQUIRED)
find_package(stb REQUIRED)

add_executable(resnet src/main.cpp)
target_compile_features(resnet PUBLIC cxx_std_20)
target_link_libraries(resnet PRIVATE onnxruntime::onnxruntime stb::stb)

# Copy resnet18-v2-7.onnx model in the executable directory
add_custom_command(TARGET resnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/resnet18-v2-7.onnx ${CMAKE_CURRENT_BINARY_DIR}/resnet18-v2-7.onnx
    COMMENT "Copying resnet18-v2-7.onnx"
)

# Copy squeezenet1.1-7.onnx model in the executable directory
add_custom_command(TARGET resnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/squeezenet1.1-7.onnx ${CMAKE_CURRENT_BINARY_DIR}/squeezenet1.1-7.onnx
    COMMENT "Copying squeezenet1.1-7.onnx"
)

# Copy labels.txt file in the executable directory
add_custom_command(TARGET resnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/models/labels.txt ${CMAKE_CURRENT_BINARY_DIR}/labels.txt
    COMMENT "Copying labels.txt"
)

# Copy images directory in the executable directory
add_custom_command(TARGET resnet POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/images ${CMAKE_CURRENT_BINARY_DIR}/images
    COMMENT "Copying images directory"
)

install(TARGETS resnet DESTINATION .)
install(FILES ${CMAKE_SOURCE_DIR}/resnet18-v2-7.onnx DESTINATION .)
install(FILES ${CMAKE_SOURCE_DIR}/squeezenet1.1-7.onnx DESTINATION .)
install(FILES ${CMAKE_SOURCE_DIR}/models/labels.txt DESTINATION .)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/images DESTINATION .)
