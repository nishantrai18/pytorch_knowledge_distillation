from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='swish_cpp',
    ext_modules=[
        CppExtension('swish_cpp', ['swish.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
