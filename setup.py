from distutils.core import setup

setup(
    name='scoupy',
    version='0.0.1dev1',
    packages=['scoupy'],
    url='https://github.com/mdomanski-usgs/scoupy',
    license='CC0 1.0',
    author='Marian Domanski',
    author_email='mdomanski@usgs.gov',
    description='Hydroacoustic sediment analysis',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python 3.6'
    ],
    python_requires='>=3',
    install_requires=['numpy', 'scipy']
)
