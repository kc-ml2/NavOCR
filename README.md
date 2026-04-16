# NavOCR

NavOCR is a lightweight navigation-oriented OCR framework for text detection and text recognition.

It is designed for robotic navigation scenarios where only navigation-relevant text should be detected, such as:

- Signboards
- Room numbers

Key points:

- Focuses on important text to avoid excessive information and improve OCR speed
- Both stand-alone and ROS 2 support
- Optimized for CPU-first robotic platforms (8 FPS on Intel CPU with OpenVINO)
- Paddle/PaddleDetection support for GPU environments


<p align="center">
    <img src="./example.svg" alt="NavOCR_example"
</p>

<p align="center">
    <img src="./NavOCR.gif" alt="NavOCR" width="572"/>
</p>


## Overview

- `navocr_standalone.py`: run detection + OCR on a single image or a directory
- `navocr/ros_node.py`: ROS 2 node entry point
- `configs/navocr_openvino.params.yaml`: OpenVINO detector + OpenVINO OCR config
- `configs/navocr_paddle.params.yaml`: PaddleDetection detector + Paddle OCR config

## Installation

### Download Model
Both the OpenVINO models and the Paddle/PaddleDetection models are included in this repository.  

```bash
git clone git@github.com:kc-ml2/NavOCR.git
```

### For OpenVINO Backend

Install OpenVINO:
```bash
pip install openvino pyyaml opencv-python numpy
```


### (Optional) For Paddle Backend

Install PaddlePaddle:
```bash
# For CPU
pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple

# For CUDA
pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Install PaddleDectetion:
```bash
git clone https://github.com/PaddlePaddle/PaddleDetection.git

cd PaddleDetection
pip install -r requirements.txt

python setup.py install
```

Install PaddleOCR:
```bash
pip install paddleocr
```

[Paddle Offical Guide](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/docs/tutorials/INSTALL.md)

### Download Testset
Not required if you use the ROS node.

```bash
# Setup python env
pip install gdown==5.2.0

# Download sample testset
mkdir data && cd data
gdown https://drive.google.com/uc?id=1GcgddRm4GsjPKUOVdmWFzeF5gElCZfx2
unzip example_sequence.zip 
cd .. && mkdir results
```

## Standalone Inference

### Run with OpenVINO backend
```bash
python navocr_standalone.py \
  --params-file configs/navocr_openvino.params.yaml \
  --infer_dir data/example_sequence/images
```

### Run with Paddle backend
```bash
python navocr_standalone.py \
  --params-file configs/navocr_paddle.params.yaml \
  --infer_dir data/example_sequence/images
```

### Single image
```bash
python navocr_standalone.py \
  --params-file configs/navocr_openvino.params.yaml \
  --input data/example_sequence/images/000000.jpg
```


## ROS 2 Node
```bash
# Build and run as ROS 2 node (detection + OCR)
cd ~/ros2_ws  # your ros workspace
colcon build --packages-select navocr
source install/setup.bash

# If you encounter oneDNN compatibility issues on CPU, set these before running:
export FLAGS_enable_pir_api=0
export FLAGS_enable_pir_in_executor=0

ros2 run navocr navocr_with_ocr_node
```

If you want to select a different params file at runtime:
```bash
ros2 run navocr navocr_with_ocr_node --ros-args \
  -p params_file:=/absolute/path/to/configs/navocr_paddle.params.yaml
```

Published topics:

- `detections_topic` default: `/navocr/detections`
- `annotated_image_topic` default: `/navocr/annotated_image`

## Training Model

Coming soon! (Dataset crawling, dataset preprocessing, model fine-tuning, ...)

## 🚧 Planned Updates
We're working on expanding support beyond store signboards detection model.
Stay tuned for upcoming features for broader navigation use cases.

- [x] Library migration due to a license issue (`ultralytics` -> `PaddleDetection`)
- [x] Alternative inference for higher FPS on CPU (Add `OpenVINO` support)
- [x] Integration with text recognition (PaddleOCR)
- [x] Integration with SLAM packages via ROS (TextMap)
- [ ] Model training scripts
- [ ] Room number and floor sign detection
- [ ] Directional guide text detection

## License
This repository is licensed under the Apache License, Version 2.0.

This project includes code and configuration files derived from
PaddleDetection (https://github.com/PaddlePaddle/PaddleDetection),
which is also licensed under the Apache License, Version 2.0.
