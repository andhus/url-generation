from setuptools import setup, find_packages

VERSION = '0.0.1'

setup(
    name='url-generation',
    version=VERSION,
    description='RNN Generation of phising/ok URL:s',
    url='https://github.com/andhus/url-generation',
    license='MIT',
    install_requires=[
        'numpy>=1.15.0',
        'pandas>=0.23.0',
        'keras>=2.2.4',
        'tensorflow>=1.12.0'
    ],
    extras_require={},
    packages=find_packages(
        exclude=['*.tests', '*.tests.*', 'tests.*', 'tests']
    ),
    tests_require=['nose']
)
