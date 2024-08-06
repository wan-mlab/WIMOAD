from setuptools import setup, find_packages

setup(
    name="WIMOAD",
    version="1.0.0",
    description="Weighted Integration of Multi-Omics data for Alzheimer’s Disease (AD) Diagnosis",
    url="https://github.com/wan-mlab/RanBALL",
    author="Hanyu Xiao, Jieqiong Wang, Shibiao Wan",
    author_email="haxiao@unmc.edu",
    license="MIT",
    packages=find_packages(where='./WIMOAD'),
    package_dir={
        '': 'WIMOAD'
    },
    include_package_data=True,
    install_requires=[
       'numpy',
        'pandas',
        'scikit-learn',
        'imbalanced-learn'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    test_suite='tests',
    python_requires=">=3.6"
)
