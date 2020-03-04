from distutils.core import setup, Extension

setup(
    name='pyprocar',
    version = "5.1.0",
    author='Francisco Munoz,Aldo Romero,Sobhit Singh,Uthpala Herath,Pedram Tavadze,Eric Bousquet, Xu He',
    author_email='fvmunoz@gmail.com,alromero@mail.wvu.edu,smsingh@mix.wvu.edu,ukh0001@mix.wvu.edu,petavazohi@mix.wvu.edu,eric.bousquet@uliege.be,mailhexu@gmail.com',
    url='https://github.com/romerogroup/pyprocar',
    download_url='https://github.com/romerogroup/pyprocar/archive/5.1.0.tar.gz',
    packages=['pyprocar', 
              'pyprocar.utilsprocar',
              'pyprocar.procarparser',
              'pyprocar.procarfilefilter',
              'pyprocar.procarplot',
              'pyprocar.procarsymmetry',
              'pyprocar.procarplotcompare',
              'pyprocar.procarunfold',
              'pyprocar.fermisurface',
              'pyprocar.procarselect',
              'pyprocar.elkparser'],
    license='LICENSE.txt',
    description='A Python library for electronic structure pre/post-processing.',
    install_requires=['seekpath>=1.0', 'numpy==1.17.2', 'scipy==1.4.1', 'matplotlib', 'ase', 'scikit-image','mayavi==4.6.2','pyfiglet'],
    scripts=['bin/procar']
)
