from distutils.core import Extension, setup

module = Extension("mykmeanssp", sources=[ "kmeans.c" ])
setup(name="mykmeanssp", version="1.0", ext_modules=[ module ])
