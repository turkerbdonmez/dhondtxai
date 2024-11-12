
from setuptools import setup, find_packages

setup(
    name='dhondtxai',
    version='0.6.2.3',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'xgboost',
        'scikit-learn',
        'catboost',
        # Random forest and AdaBoost are part of scikit-learn
    ],
    description="A library for applying D'Hondt method on feature importances from decision tree models",
    author='Türker Berk DÖNMEZ',
    author_email='furkan@example.com',
    url='https://github.com/turkerbdonmez/dhondtxai',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
