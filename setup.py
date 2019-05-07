# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 13:02:44 2019

@author: Chris
"""

from setuptools import setup, find_packages
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    setup(
        name='compmatscipy',
        version='0.0.1',
        description='facilitating efficient computational materials research',
        long_description=open(os.path.join(module_dir, 'README.md')).read(),
        url='https://github.com/CJBartel/compmatscipy',
        author=['Christopher J. Bartel'],
        author_email=['christopher.bartel@colorado.edu'],
        license='MIT',
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        install_requires=[],
        extras_require={},
        classifiers=[],
        test_suite='',
        tests_require=[],
        scripts=[]
    )
    
#        package_data={'modules' : ['module_data/*.json',
#                                   'module_data/*.p']}