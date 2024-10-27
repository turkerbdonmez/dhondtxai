
from setuptools import setup, find_packages

setup(
    name='dhondt-xai',
    version='0.1',
    packages=find_packages(include=['dhondt_xai', 'dhondt_xai.*']),
    install_requires=['numpy', 'matplotlib', 'xgboost'],
    author='Your Name',
    author_email='your.email@example.com',
    description='A library for applying D'Hondt method to feature importances in decision tree models.',
    url='https://github.com/yourusername/dhondt-xai',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
