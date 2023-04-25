from setuptools import setup

setup(
    name='radar',
    version='0.1.0',
    description='radar package',
    url='https://github.com/cbernard10/radar',
    author='Clément Bernard',
    author_email='clement.bernard@protonmail.com',
    license='MIT',
    packages=['radarpkg2'],
    install_requires=['numpy', 'scipy', 'umap-learn[plot]', 'hdbscan', 'numba', 'scikit-learn', 'opencv-python']
)