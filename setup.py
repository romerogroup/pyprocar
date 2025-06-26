from setuptools import setup

setup(
    package_data={"": ["setup.json", '*.ini','*.yml']},
    scripts=["scripts/procar.py"],
    )