#!/usr/bin/env python
"""
RealityGuard: AI-Powered Real-Time Privacy System
Setup script for package installation
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core requirements
requirements = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "ultralytics>=8.0.0",
    "opencv-python>=4.8.0",
    "numpy>=1.24.0",
    "pillow>=10.0.0",
    "tqdm>=4.65.0",
]

# Optional requirements for different features
extras = {
    "gpu": [
        "tensorrt>=8.6.0",
        "cuda-python>=12.0",
    ],
    "streaming": [
        "aiortc>=1.5.0",
        "av>=10.0.0",
    ],
    "web": [
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "gunicorn>=21.0.0",
    ],
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
    ],
}

setup(
    name="realityguard",
    version="1.0.0",
    author="Chinmay Shrivastava",
    author_email="cshrivastava2000@gmail.com",
    description="World's first real-time AI replacement privacy system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JonSnow1807/RealityGuard",
    packages=find_packages(exclude=["tests", "test_*", "docs"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require=extras,
    entry_points={
        "console_scripts": [
            "realityguard=realityguard.cli:main",
            "realityguard-server=realityguard.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "realityguard": ["models/*.pt", "configs/*.yaml"],
    },
    keywords="privacy ai computer-vision real-time video-processing gan synthetic-data",
    project_urls={
        "Bug Reports": "https://github.com/JonSnow1807/RealityGuard/issues",
        "Source": "https://github.com/JonSnow1807/RealityGuard",
        "Documentation": "https://github.com/JonSnow1807/RealityGuard/blob/main/docs/API.md",
    },
)