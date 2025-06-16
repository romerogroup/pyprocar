from setuptools import setup
from setuptools_scm import ScmVersion

def version_for_project(version: ScmVersion) -> str:
   return str(version.tag)

setup(use_scm_version={'version_scheme':version_for_project},
    package_data={"": ["setup.json", '*.ini','*.yml']},
    scripts=["scripts/procar.py"],
    )