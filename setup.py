from setuptools import setup

metadata = dict(
    name="peal",
    version="0.0.5",
    author="alienkrieg",
    author_email="alienkrieg@gmail.com",
    description="Python Package for Evolutionary Algorithms",
    packages=[
        "peal",
        "peal.core",
        "peal.operations",
    ],
    ext_package="",
    ext_modules=[],
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.21.1",
    ],
    python_requires=">=3.9, <3.10",
)

if __name__ == "__main__":
    setup(**metadata)
