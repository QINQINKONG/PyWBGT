from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
	"WBGT",
        ["WBGT.pyx"],
        libraries=["m"],
        extra_compile_args=['-qopenmp','-Ofast'],
        extra_link_args=['-qopenmp'],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name='WBGT',
    ext_modules=cythonize(ext_modules,annotate=True),
    zip_safe=False,
    include_dirs=[numpy.get_include()]
)
