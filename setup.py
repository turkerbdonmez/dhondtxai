
from pathlib import Path

from setuptools import setup, find_packages


ROOT = Path(__file__).resolve().parent
VERSION = "0.9.5.6"
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name='dhondtxai',
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
    ],
    extras_require={
        'sklearn': ['scikit-learn'],
        'xgboost': ['xgboost'],
        'lightgbm': ['lightgbm'],
        'catboost': ['catboost'],
        'torch': ['torch'],
        'all-models': ['scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'torch'],
        'dev': ['pytest', 'scikit-learn', 'build', 'twine'],
    },
    description="D'Hondt-projected removal-effect attributions for tabular XAI",
    long_description=README,
    long_description_content_type="text/markdown",
    author='Turker Berk Donmez',
    author_email='turkerberkdonmez@yahoo.com',
    url='https://github.com/turkerbdonmez/dhondtxai',
    license='MIT',
    keywords=[
        "explainable-ai",
        "xai",
        "model-agnostic",
        "feature-attribution",
        "dhondt",
        "tabular",
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: OS Independent',
    ],
    project_urls={
        'Source': 'https://github.com/turkerbdonmez/dhondtxai',
        'Paper': 'https://doi.org/10.48550/arXiv.2411.05196',
    },
    python_requires='>=3.8',
)
