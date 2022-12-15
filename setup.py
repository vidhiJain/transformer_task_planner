from setuptools import setup, find_packages

__author__ = "Vidhi Jain"
__copyright__ = "2022, CMU"


setup(
    name="temporal_task_planner",
    author="Vidhi Jain",
    author_email="vidhij@andrew.cmu.edu",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
