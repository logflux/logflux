# Copyright 2024 Guineng Zheng and The University of Utah
# All rights reserved.
# This file is part of the logflux project, which is licensed under the
# simplified BSD license.

import setuptools

setuptools.setup(
    name="logflux",
    version="1.0.0",
    author="logflux_author",
    author_email="logflux@github.com",
    description="logflux log parsing toolkit",
    long_description ="logflux log parsing toolkit",
    packages=setuptools.find_packages(where="src"),
    package_dir={"":"src"},
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=["numpy", "scikit-learn", "torch"],
    classifiers=(
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ),
    license="Apache-2.0 License",
    keywords=['log analysis', 'log parsing']
)
