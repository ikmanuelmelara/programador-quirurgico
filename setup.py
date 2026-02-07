"""
Setup script for Programador Quirúrgico Inteligente
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="programador-quirurgico",
    version="4.9.0",
    author="Tu Nombre",
    author_email="tu@email.com",
    description="Sistema de optimización para programación quirúrgica con ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TU_USUARIO/programador-quirurgico",
    project_urls={
        "Bug Tracker": "https://github.com/TU_USUARIO/programador-quirurgico/issues",
        "Documentation": "https://github.com/TU_USUARIO/programador-quirurgico/wiki",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "advanced": [
            "ortools>=9.4.0",
            "deap>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "programador-quirurgico=main:demo_interactiva",
        ],
    },
)
