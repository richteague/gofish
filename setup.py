import setuptools

setuptools.setup(
    name="gofish",
    version="1.3.3",
    author="Richard Teague",
    author_email="richard.d.teague@cfa.harvard.edu",
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
      ]
)
