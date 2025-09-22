from setuptools import setup, find_packages

setup(
    name="triton_kernels",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "triton>=2.0.0",
        "torch>=2.0.0",
        "numpy",
    ],
    author="Kernelize AI",
    description="Custom Triton kernels for high-performance computing",
)