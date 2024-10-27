from setuptools import setup, find_packages

setup(
    name='dhondt-xai',
    version='1.0.0',
    description='D'Hondt method for feature importance in tree-based models',
    author='Ali Furkan Kamanlı',
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
