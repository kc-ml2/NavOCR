from setuptools import find_packages, setup

package_name = 'navocr'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            'share/' + package_name + '/configs',
            [
                'configs/navocr_openvino.params.yaml',
                'configs/navocr_paddle.params.yaml',
            ],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='KC-ML2',
    maintainer_email='contact@kc-ml2.com',
    description='NavOCR - Sign detection and OCR using OpenVINO and PaddleOCR',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'navocr_with_ocr_node = navocr.ros_node:main',
        ],
    },
)
