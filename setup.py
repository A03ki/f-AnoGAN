from setuptools import setup


setup(
    name="fanogan",
    version="0.0.1",
    description="Test f-AnoGAN by using datasets.",
    author="A03ki",
    python_requires=">=3.6",
    install_requires=["torch", "torchvision", "torchvision4ad", "matplotlib",
                      "numpy", "pandas", "Pillow", "scikit-learn"],
    url="https://github.com/A03ki/f-AnoGAN",
    license="MIT License",
    packages=["fanogan"]
)
