## copied from the docs:
##   http://docs.ros.org/en/api/catkin/html/howto/format2/installing_python.html#modules

## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['yolov3_pytorch_ros', 'utils', 'models'],
    package_dir={'': 'src'})

setup(**setup_args)
