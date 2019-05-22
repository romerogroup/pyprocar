from distutils.core import setup, Extension
import pyprocar 

setup(
    name='pyprocar',
    version = pyprocar.__version__,
    author='Francisco Munoz,Aldo Romero,Sobhit Singh,Uthpala Herath,Pedram Tavadze,Eric Bousquet,Xu He',
    author_email='fvmunoz@gmail.com,alromero@mail.wvu.edu,smsingh@mix.wvu.edu,ukh0001@mix.wvu.edu,petavazohi@mix.wvu.edu,eric.bousquet@uliege.be,mailhexu@gmail.com',
    url='https://github.com/romerogroup/pyprocar',
    download_url='https://github.com/romerogroup/pyprocar/archive/3.5.tar.gz',
    packages=['pyprocar', 
              'pyprocar.utilsprocar',
              'pyprocar.fermisurface',
              'pyprocar.procarparser',
              'pyprocar.procarfilefilter',
              'pyprocar.procarplot',
              'pyprocar.procarsymmetry',
              'pyprocar.procarplotcompare',
              'pyprocar.procarunfold',
              'pyprocar.procarselect'],
    license='LICENSE.txt',
    description='A python package for analyzing PROCAR files obtained from VASP and Abinit.',
    install_requires=['seekpath>=1.0'],
)
