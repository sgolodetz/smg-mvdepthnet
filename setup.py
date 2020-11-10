from setuptools import find_packages, setup

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setup(
    name="smg-mvdepthnet",
    version="0.0.1",
    author="Stuart Golodetz",
    author_email="stuart.golodetz@cs.ox.ac.uk",
    description="Wrapper for MVDepthNet",
    long_description="",  #long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sgolodetz/smg-mvdepthnet",
    packages=find_packages(include=["smg.mvdepthnet", "smg.external.*"]),
    include_package_data=True,
    install_requires=[
        "matplotlib",
        "numpy",
        "opencv-contrib-python==3.4.2.16",
        "torch==1.7.0+cu101",
        "torchaudio===0.7.0",
        "torchvision==0.8.1+cu101"
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
