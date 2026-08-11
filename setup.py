import os

from setuptools import find_packages, setup

package_name = "navocr"


def collect_data_files(source_dir: str, install_base: str) -> list:
    data_files = []
    for root, _dirs, files in os.walk(source_dir):
        if not files:
            continue
        rel_dir = os.path.relpath(root, source_dir)
        target_dir = install_base if rel_dir == "." else os.path.join(install_base, rel_dir)
        file_paths = [os.path.join(root, name) for name in files]
        data_files.append((target_dir, file_paths))
    return data_files


data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
]
data_files += collect_data_files("configs", "share/" + package_name + "/configs")
data_files += collect_data_files("model", "share/" + package_name + "/model")

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=data_files,
    install_requires=["setuptools"],
    extras_require={
        # Dataset preprocessing pipeline (CLIP filter, OCR matching, COCO export).
        # Note: paddlepaddle/paddleocr are not pip-installed here — install them
        # manually with the right CUDA variant for your hardware. See README. #TODO: readme update
        "preprocess": [
            "transformers>=4.30",
            "torch",
            "Pillow",
            "textdistance",
            "pyyaml",
        ],
        # DeepL translation fallback for cross-language store-name matching.
        "translate": ["deepl"],
    },
    zip_safe=True,
    maintainer="KC-ML2",
    maintainer_email="contact@kc-ml2.com",
    description="NavOCR - Sign detection and OCR using OpenVINO and PaddleOCR",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "navocr_with_ocr_node = navocr.ros_node:main",
            "preprocess_navocr = navocr.preprocess.runner:main",
        ],
    },
)
