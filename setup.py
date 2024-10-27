from setuptools import setup, find_packages

setup(
    name='dhondt-xai',
    version='1.0.0',
    description="DHondt method for feature importance in tree-based models",
    author='Türker Berk Dönmez',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'xgboost',
        'catboost',
        'scikit-learn',
        'pandas',
    ],
)
