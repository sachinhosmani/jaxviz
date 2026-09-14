from setuptools import find_packages, setup


setup(
    name="jaxviz",
    version="0.1.4",
    description="Interactive JAX forward-pass visualizer for notebooks",
    license="GPL-3.0-only",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "jax>=0.4.0",
        "ipython>=7.0.0",
    ],
    extras_require={
        "test": ["pytest>=7.0", "flax>=0.8.0"],
    },
    python_requires=">=3.9",
    package_data={
        "jaxviz": ["templates/*.html", "assets/*.css", "assets/*.js"],
    },
    keywords=["jax", "visualization", "neural-network", "sharding", "spmd"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
