import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyGNMF",
    version="1.0.6",
    author=["Nirav Lekinwala","Mani Bhushan"],
    author_email="nirav.lekinwala@gmail.com",
    description="Python implementation of Generalised Non-negative Matrix Factorisation with Multiplicative and Projected Gradient Approaches.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/niravl/pyGNMF",
    project_urls={
        "Bug Tracker": "https://github.com/niravl/pyGNMF/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = ['numpy',
                        'tqdm'],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)
