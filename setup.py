from distutils.core import setup

setup(
    name='cbar',
    version='0.1',
    packages=['cbar'],
    license='MIT',
    install_requires=[
        'Click',
        'nltk>=3',
        'numpy',
        'pandas>0.18',
        'requests',
        'scipy',
        'scikit-learn<0.18'
    ],
    entry_points='''
        [console_scripts]
        cbar=cbar.main:cli
    ''',
    description=('A Python package for content-based audio retrieval with '
                 'text queries.'),
    author='David Schwertfeger',
    author_email='david.schwertfeger@gmail.com',
    url='https://github.com/dschwertfeger/cbar',
    download_url='https://github.com/dschwertfeger/cbar/tarball/0.1',
    keywords=['content', 'based', 'audio', 'retrieval'],
    classifiers=[],
)
