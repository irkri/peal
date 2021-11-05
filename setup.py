from setuptools import setup

metadata = dict(
    name="peal",
    version="0.0.1",
    author="alienkrieg",
    author_email="alienkrieg@gmail.com",
    description="A Python Package for Evolutionary Algorithms",
    packages=[
        "peal",
        "peal.environment",
        "peal.individual",
        "peal.evaluation",
        "peal.operations",
        "peal.operations.mutation",
        "peal.operations.reproduction",
        "peal.operations.selection",
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
