from setuptools import setup, find_packages

setup(
    name='structml',
    version='0.1',
    packages=find_packages(),
description='A package for working with data using NLP and ML.',
    author='PocketDoc Labs',
    author_email='visuallyadequate@gmail.com',
    url='https://github.com/PocketDocLabs/struct-ml',
    install_requires=[
        'transformers',
        'torch',
        'rich'
    ],
)