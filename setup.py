"""
Setup script for the Job Description Parser package.
"""

from setuptools import setup, find_packages
import os

def read_file(filename):
    """Read file content safely."""
    try:
        with open(filename, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return ""

def read_requirements(filename):
    """Read requirements from file."""
    try:
        with open(filename, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return []

long_description = read_file("README.md")
if not long_description:
    long_description = "AI-driven job description parser that extracts structured information from unstructured job postings"

requirements = read_requirements("requirements.txt")

setup(
    name="job-description-parser",
    version="1.0.0",
    author="Job Description Parser Team",
    author_email="support@example.com",
    description="AI-driven job description parser that extracts structured information from unstructured job postings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/job-description-parser",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/job-description-parser/issues",
        "Documentation": "https://github.com/yourusername/job-description-parser/wiki",
        "Source Code": "https://github.com/yourusername/job-description-parser",
        "Changelog": "https://github.com/yourusername/job-description-parser/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests*", ".kiro*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="nlp job-parsing machine-learning text-analysis hr-tech semantic-matching ner",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.6.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "job-parser=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "job_parser": ["*.csv", "*.json"],
    },
    zip_safe=False,
)