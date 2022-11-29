from setuptools import setup

setup(
    name='pylag',
    version='2.0',
    license='MIT',
    author='Dan Wilkins',
    author_email='dan.wilkins@stanford.edu',
    packages=['pylag'],
    url='https://github.com/wilkinsdr/pylag',
    description='An object-oriented and (hopefully) easy to use X-ray timing analysis package',
    install_requires=['numpy',
    'scipy',
    'astropy']
)
