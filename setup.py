from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setup(
    name='ABlooper',
    version='1.1.1',
    description='Set of functions to predict CDR structure',
    license='BSD 3-clause license',
    maintainer='Brennan Abanades',
    long_description=long_description,
    long_description_content_type='text/markdown',
    maintainer_email='brennan.abanadeskenyon@stx.ox.ac.uk',
    include_package_data=True,
    packages=find_packages(include=('ABlooper', 'ABlooper.*')),
    entry_points={'console_scripts': ['ABlooper=ABlooper.command_line:main']},
    install_requires=[
        'numpy',
        'einops>=0.3',
        'torch>=1.6',
    ],
)
