"""Package installer."""
from setuptools import find_packages, setup
LONG_DESCRIPTION = '''
A parallel COREX based on GPU computation with an adaptive Monte Carlo sampling.
gpuCOREX enables batch computation and moves all nain computation procedures
(SASA, Thermodynamics Quantities, Sampling) to GPU with a short precomputation
procedure on CPUs. Adaptive Monte Carlo sampling allows a auto adaptive sampling
threshold to balance output quality and computation efficiency.
We achieve further acceleration and higher COREX quality than pCOREX.
'''
setup(
    name='gpucorex',
    version='0.0.2',
    description='The COREX Algorithm Run on GPUs',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author='Jiarui Li, Ramgopal Mettu, Samuel Landry',
    author_email=('jli78@tulane.edu', 'rmettu@tulane.edu', 'landry@tulane.edu'),
    url='https://git.tulane.edu/apl/apl/-/tree/COREX-BEST/COREX/gpuCOREX',
    license='Apache 2.0',
    install_requires=[
        'Cython',
        'torch',
        'termcolor',
        'pandas',
        'numpy==1.24.4',
        'biopandas==0.4.1',
        'biotite==0.38.0',
        'scipy==1.10.1',
        'tqdm==4.66.1',
        'scikit-learn==1.1.2',
        'matplotlib==3.7.3'
    ],
    entry_points={
        'console_scripts': [
            'gpucorex = gpucorex.main:main_gpucorex_',
        ],
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache 2.0 License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=find_packages('.'),
    include_package_data = True
)
