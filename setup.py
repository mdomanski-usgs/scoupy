from setuptools import setup

from sphinx.setup_command import BuildDoc

name = 'scoupy'
version = '0.0'
release = '0.0.0'
cmdclass = {'build_sphinx': BuildDoc}
docs_source = 'docs/'
docs_build_dir = 'docs/_build'
docs_builder = 'html'

setup(
    name=name,
    version=release,
    packages=['scoupy'],
    url='https://github.com/mdomanski-usgs/scoupy',
    test_suite = 'test_scoupy.py',
    license='CC0 1.0',
    author='Marian Domanski',
    author_email='mdomanski@usgs.gov',
    description='Sediment acoustic analysis in Python',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python 3.7'
    ],
    python_requires='>=3',
    install_requires=['numpy', 'scipy'],
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release),
            'source_dir': ('setup.py', docs_source),
            'build_dir': ('setup.py', docs_build_dir),
            'builder': ('setup.py', docs_builder)}
    }
)
