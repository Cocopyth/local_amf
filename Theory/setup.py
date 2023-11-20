from setuptools import setup, Extension

module = Extension('intersection',
                   sources=['intersection.c'])

setup(name='Intersection',
      version='1.0',
      description='Line segment intersection module',
      ext_modules=[module])