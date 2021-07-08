from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
	"coszenith",
        ["coszenith.pyx"],
        extra_compile_args=['-qopenmp','-Ofast'],
        extra_link_args=['-qopenmp'],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name='coszenith',
    ext_modules=cythonize(ext_modules,annotate=True),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
