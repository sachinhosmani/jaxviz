from setuptools import setup, find_packages

setup(
    name='jaxtrace',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'jax>=0.4.0',
        'ipython>=7.0.0',
    ],
    extras_require={
        'test': ['pytest>=7.0', 'flax>=0.8.0'],
    },
    python_requires='>=3.9',
    package_data={
        'jaxtrace': ['templates/*.html', 'assets/*'],
    },
    description="Interactive JAX model forward-pass visualizer for notebooks",
    long_description=(
        "jaxtrace displays an interactive graph of the forward pass of a JAX model "
        "directly in the notebook with a single line of code. The computation graph is "
        "obtained from the model's jaxpr and rendered as a draggable, zoomable, "
        "collapsible graph with tensor shapes on every edge."
    ),
    long_description_content_type='text/plain',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
)
