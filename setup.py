import setuptools

setuptools.setup(
    name="gofish",
    version="0.1.2",
    author="Richard Teague",
    author_email="rteague@umich.edu",
    description="Fishing for molecular line emission in protoplanetary disks.",
    url="https://github.com/richteague/gofish",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "astropy",
        "scipy",
        "astro-eddy",
      ]
)
