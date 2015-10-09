from setuptools import setup, find_packages

setup(
    name = 'homotopy',
    version = 0.1,

    packages = find_packages(),
    requires = ['numpy'],

    # metadata for upload to PyPI
    author = "Shifan Zuo",
    author_email = "sfzuo@bao.ac.cn",
    description = "Homotopy algorithm for l1-norm minimization",
    license = "GPL v3.0",
    url = "http://github.com/zuoshifan/homotopy"
)
