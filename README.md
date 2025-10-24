# NavOCR

## Text Detection for Navigation!

NavOCR is an open-source project that provides a lightweight text detection model for navigation.  
Other publicly available OCR models often work too well and detect texts unrelated to navigation, such as advertisements, logos, or price tags.
NavOCR detects only the text that is necessary for navigation, such as signboards, directional guides, and room numbers.  
We provide the full pipeline for model training (including data crawling, dataset preprocessing, and fine-tuning).

❗This repository is currently under heavy refactoring and development.
Please note that it may contain unstable components.
Improvements and updates will be released soon.

<p align="center">
    <img src="./example.svg" alt="NavOCR_example"
</p>

<p align="center">
    <img src="./NavOCR.gif" alt="NavOCR" width="572"/>
</p>


## How to Use

### Model Download
Our model is included in this repo. So, clone this repo to download the model!  
The current model supports detection of store signboards only.
Detection of other navigation-relevant text types will be supported in future updates.

```bash
git clone git@github.com:kc-ml2/NavOCR.git
```

### Setup
```bash
# Setup python env
pip install -r requirements.txt

# Download sample testset
mkdir data && cd data
gdown https://drive.google.com/uc?id=1GcgddRm4GsjPKUOVdmWFzeF5gElCZfx2
unzip example_sequence.zip 
cd .. && mkdir results
```

### Run NavOCR!
```bash
# Fix the directory of RUN_DIR in env.sh first
./run_nav_ocr.sh
```

## Training Model

Coming soon! (Dataset crawling, dataset preprocessing, model fine-tuning, ...)

## 🚧 Planned Updates
We're working on expanding support beyond store signboards detection model.
Stay tuned for upcoming features for broader navigation use cases.

- Model training scripts
- Integration with text recognition (Only detection is available now.)
- Room number and floor sign detection
- Directional guide text detection
- Integration with other SLAM packages via ROS

