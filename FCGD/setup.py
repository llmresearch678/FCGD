"""
Setup configuration for the FCGD package.
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith('#')]

setup(
    name="fcgd",
    version="1.0.0",
    author="Usman Ahmad Usmani, Arunava Roy, Junzo Watada",
    author_email="usmanahmad.usmani@xmu.edu.my",
    description=(
        "Frequency-Conditioned Graph Diffusion for Robust Unsupervised "
        "Domain Adaptation in Medical Image Segmentation"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/usmanusmani/FCGD",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    entry_points={
        "console_scripts": [
            "fcgd-train=scripts.train:main",
            "fcgd-eval=scripts.evaluate:main",
            "fcgd-infer=scripts.infer:main",
        ]
    },
)
