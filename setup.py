from setuptools import setup, find_packages

setup(
    name='ABlooper',
    version='0.0.1',
    description='Set of functions to predict CDR structure',
    license='BSD 3-clause license',
    maintainer='Brennan Abanades',
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
