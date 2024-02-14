from setuptools import setup, find_packages

setup(
    name='structml',
    version='0.0.3',
    packages=find_packages(),
description='A package for working with data using NLP and ML.',
    author='PocketDoc Labs',
    author_email='visuallyadequate@gmail.com',
    url='https://github.com/PocketDocLabs/struct-ml',
    install_requires=[
        'transformers >= 4.37.0',
        'torch >= 2.0.0',
        'rich',
    ],
)