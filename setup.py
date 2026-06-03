from setuptools import setup, find_packages

setup(
    name="eiscirc",
    version="0.1.0",
    description="Flexible equivalent-circuit impedance modelling and fitting for EIS",
    long_description=open("README.md", encoding="utf-8").read() if True else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas"
    ],
    python_requires='>=3.8',
)