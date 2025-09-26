from setuptools import Extension, setup


module = Extension("symnmf_c", sources=['symnmfmodule.c'])
setup(name='symnmf_c',
        version='1.0',
        description='Python wrapper from custom C extension',
        ext_modules=[module])