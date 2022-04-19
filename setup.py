"""Package setup."""
import io
import re

from setuptools import find_packages, setup

match = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open("terminator/__init__.py", encoding="utf_8_sig").read(),
)
if match is None:
    raise SystemExit("Version number not found.")
__version__ = match.group(1)

setup(
    name="terminator",
    version=__version__,
    author="IBM Resarch team",
    author_email=["jannis.born@gmx.de, drugilsberg@gmail.com"],
    packages=find_packages(),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    package_data={"terminator": ["py.typed"]},
    install_requires=["transformers", "numpy", "tqdm", "selfies==1.0.4", "modlamp"],
)
