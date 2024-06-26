import os
from distutils.core import setup
from setuptools import find_packages
import json
from pathlib import Path
from typing import Optional


def get_version_info():
    """ Retrieve version info from setup.json.
    """
    base_path = Path(__file__).parent.absolute()
    base_path /= 'setup.json'
    with open(base_path.as_posix(), 'r') as rf:
        release_data = json.load(rf)
    return release_data

def write_version_py():
    """ Writes pyprocar/version.py.
    """
    base_path = Path(__file__).parent.absolute()
    filename = base_path / 'pyprocar' / 'version.py'
    release_data = get_version_info() 
    version_info_string = "# THIS FILE IS GENERATED FROM PYPROCAR SETUP.PY."
    for key in release_data:
        version_info_string += f"\n{key} = '{release_data[key]}'"
    with open(filename.as_posix(), 'w') as wf:
        wf.write(version_info_string)
    return release_data

# Function to read the requirements.txt file
def read_requirements(file_name):
    with open(file_name) as f:
        return f.read().splitlines()

# Path to the requirements.txt file
requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')

# Read the requirements from the requirements.txt file
requirements = read_requirements(requirements_path)


data = write_version_py()

setup(
    name=data["name"],
    version=data["version"],
    description=data["description"],
    author=data["author"],
    author_email=data["email"],
    url=data["url"],
    download_url=data["download_url"],
    license="LICENSE",
    install_requires=requirements,
    data_files=[("", ["LICENSE"])],
    package_data={"": ["setup.json", '*.ini','*.yml']},
    scripts=["scripts/procar.py"],
    packages=find_packages(exclude=["scripts"]),
)
