from setuptools import setup, find_packages

setup(
    name="renaissance_text_gan",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.5",
        "Pillow>=8.2.0",
        "tqdm>=4.61.0",
        "matplotlib>=3.4.2",
        "scikit-image>=0.18.1",
        "opencv-python>=4.5.2",
        "lpips>=0.1.4",
        "tensorboard>=2.6.0",
        "requests>=2.25.1",
    ],
    author="Renaissance Text GAN Team",
    author_email="example@example.com",
    description="A GAN for generating synthetic Renaissance-style text with printing imperfections",
    keywords="gan, deep learning, renaissance, text generation",
    python_requires=">=3.6",
) 