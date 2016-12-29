from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='ANN',
      version='0.1',
      description='This is an implementation of an artificial neural network class, which is meant to provide low level classes for building feedforward, recurrent, convolutional and other kinds of experimental neural networks',
      long_description=readme(),
      url='https://github.com/AbstractGeek/rusmalai-ncbs.git/libraries/ANN',
      author='RUSMALAI',
      author_email='sahil.moza@gmail.com',
      license='MIT',
      packages=['ANN'],
      install_requires=[ 'numpy' ],
      zip_safe=False)
