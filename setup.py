from distutils.core import setup, Extension
import pyprocar3 

setup(
    name='pyprocar3',
    version = pyprocar3.__version__,
    author='Francisco Munoz,Aldo Romero,Sobhit Singh,Uthpala Herath,Pedram Tavadze,Eric Bousquet,Xu He',
    author_email='fvmunoz@gmail.com,alromero@mail.wvu.edu,smsingh@mix.wvu.edu,ukh0001@mix.wvu.edu,petavazohi@mix.wvu.edu,eric.bousquet@uliege.be,mailhexu@gmail.com',
    url='https://github.com/uthpalah/PyProcar',
    download_url='https://github.com/uthpalah/PyProcar/archive/3.0.tar.gz',
    packages=['pyprocar3', 
              'pyprocar3.utilsprocar',
              'pyprocar3.fermisurface',
              'pyprocar3.procarparser',
              'pyprocar3.procarfilefilter',
              'pyprocar3.procarplot',
              'pyprocar3.procarsymmetry',
              'pyprocar3.procarplotcompare',
              'pyprocar3.procarselect'],
    license='LICENSE.txt',
    description='A python package for analyzing PROCAR files obtained from VASP and Abinit.',
    #long_description=open('README.rst','rt').read(),
#    install_requires=[
#        "numpy >= 1.5",
#        "scipy >= 0.9",
#    ],
)
