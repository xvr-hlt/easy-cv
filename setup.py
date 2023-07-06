from io import open

from setuptools import find_packages, setup

with open('easy_cv/__init__.py', 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split('=')[1].strip(' \'"')
            break
    else:
        version = '0.0.1'

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

REQUIRES = [
    'numpy==1.17.2',
    'scipy==1.10.0',
    'torch==1.2.0'
]

setup(
    name='easy-cv',
    version=version,
    description='',
    long_description=readme,
    author='Xavier Holt',
    author_email='holt.xavier@gmail.com',
    maintainer='Xavier Holt',
    maintainer_email='holt.xavier@gmail.com',
    url='https://github.com/xvr-hlt/easy-cv',
    install_requires=REQUIRES,
    packages=find_packages(),
)
