import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
     name='cinnabar',
     version='0.1',
     scripts=[],
     author="Tommaso Zoppi",
     author_email="tommaso.zoppi@unitn.it",
     description="prediCtIoN rejectioN strAtegies for Binary and multi-class clAssifieRs",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/tommyippoz/CINNABAR",
     keywords=['machine learning', 'confidence', 'prediction rejection', 'omission'],
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
)
