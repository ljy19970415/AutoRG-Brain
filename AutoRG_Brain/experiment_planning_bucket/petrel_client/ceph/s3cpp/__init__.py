from os import path
from ctypes import cdll

root_path = path.dirname(__file__)
libs_path = path.join(root_path, 'libs')

libs = [
    'libaws-c-common.so',
    'libaws-checksums.so',
    'libaws-c-event-stream.so',
    'libaws-cpp-sdk-core.so',
    'libaws-cpp-sdk-s3.so'
]

for lib in libs:
    lib_path = path.join(libs_path, lib)
    cdll.LoadLibrary(lib_path)
