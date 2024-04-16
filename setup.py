from setuptools import find_packages
from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    author="Wufei Ma",
    author_email="wufeim@gmail.com",
    name="imagenet3d",
    version="0.1.0",
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3"
    ],
    description="ImageNet3D experiments.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    packages=find_packages(include=["imagenet3d", "imagenet3d.*"]),
    url="https://github.com/wufeim/imagenet3d_exp",
    zip_safe=False,
)
