import os
import torch
# from torch.utils.ffi import create_extension
from torch.utils.cpp_extension import BuildExtension, CppExtension
from setuptools import setup

sources = []
headers = []
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/psroi_pooling_cuda.c']
    headers += ['src/psroi_pooling_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['src/cuda/psroi_pooling.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = CppExtension(
    '_ext.psroi_pooling',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

print("compiled!",ffi)

if __name__ == '__main__':

    setup(
        name='psroi_pooling',
        ext_modules = [ffi],
        cmdclass={'build_ext': BuildExtension}
    )
    # ffi.build()
