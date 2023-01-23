from setuptools import find_packages,setup
from typing import List 

HYPEN_E_DOT='-e .'
REQUIREMENTS_FILE_NAME='requirements.txt'

def get_requirements()->List[str]:
    with open(REQUIREMENTS_FILE_NAME) as requirements_file:
        requirements_list = requirements_file.readlines()
    requirements_list = [requirement_name.replace('\n',"") for requirement_name in requirements_list]
    if HYPEN_E_DOT in requirements_list:
        requirements_list.remove(HYPEN_E_DOT)
    return requirements_list

setup(
    name='flightfare',
    version='0.0.1',
    author='gopal',
    author_email='gopalakrishna9101997@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)