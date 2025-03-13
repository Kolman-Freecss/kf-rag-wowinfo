from setuptools import setup, find_packages

setup(
    name='kf_rag_wowinfo',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'chromadb',
        'sentence-transformers',
        'google-generativeai',
    ],
)
