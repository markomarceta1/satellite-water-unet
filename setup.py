from setuptools import find_packages, setup

setup(
    name="swu",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "matplotlib",
        "pillow",
        "numpy",
    ],
)
