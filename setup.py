import setuptools

DESCRIPTION = (
    "Package for approximating solutions to differential equations"
    " with neural networks based on TensorFlow"
)


with open("requirements/main.txt", "r") as file_handler:
    REQUIREMENTS = file_handler.read().split("\n")
    REQUIREMENTS = [x for x in REQUIREMENTS if not x.startswith("#")]


setuptools.setup(
    name="xidiff",
    version="0.0.1",
    author="Dmytro Lukashov",
    author_email="dimonluk2.0@gmail.com",
    description=DESCRIPTION,
    url="https://github.com/DimonLuk/xidiff",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=["xidiff"],
    python_requires=">=3.8",
    data_files=[("requirements", ["requirements/main.txt"])],
    package_data={"": ["requirements/main.txt"]},
    install_requires=REQUIREMENTS,
)
