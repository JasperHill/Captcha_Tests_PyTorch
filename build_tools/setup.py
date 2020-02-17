from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='custom_layers_cpp',
      ext_modules=[cpp_extension.CppExtension('custom_layers_cpp', ['Custom_Layers.cpp'])],
      cmdclass={'build_ext' : cpp_extension.BuildExtension})
