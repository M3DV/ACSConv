from setuptools import setup, find_packages

import acsconv


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

with open('requirements.txt', 'r') as f:
    requirements = f.readlines()

setup(
    name='ACSConv',
    version=acsconv.__version__,
    url='https://github.com/M3DV/ACSConv',
    license='Apache-2.0 License',
    author='Jiancheng Yang and Xiaoyang Huang',
    author_email='jekyll4168@sjtu.edu.cn',
    description='[IEEE JBHI] Reinventing 2D Convolutions for 3D Images',
    long_description=readme(),
    install_requires=requirements,
    packages=find_packages(),
    zip_safe=True
)