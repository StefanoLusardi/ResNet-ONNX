import requests


def download_file(url, output_path):
    try:
        # Send GET request to the raw URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise HTTPError for bad responses

        # Write the file content to the output path
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded successfully: {output_path}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    model_name = "squeezenet1.1-7.onnx"
    print("Downloading: ",model_name)
    url = "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx?download="
    download_file(url, model_name)

    model_name = "resnet18-v2-7.onnx"
    print("Downloading: ",model_name)
    url = "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx?download="
    download_file(url, model_name)
