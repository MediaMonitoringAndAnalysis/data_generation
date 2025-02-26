from setuptools import setup, find_packages

setup(
    name="data_generation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pandas",
        "openai-multiproc-inference",
    ],
    author="Reporter.ai",
    author_email="reporter.ai@boldcode.io",
    description="A library for data generation using RAG techniques",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)