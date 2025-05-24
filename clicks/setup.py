from setuptools import setup, find_packages

setup(
    name="clicks",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests>=2.31.0",
        "pillow>=10.0.0",
        "tenacity>=8.2.3",
        "colorama>=0.4.6",
        "tqdm>=4.66.1",
        "pandas>=2.0.0",
        "scipy>=1.11.1",
        "anthropic>=0.18.1",
        "openai>=1.66.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "google-genai>=0.6.0",
        "generalagents>=0.1.0",
        "SpeechRecognition",
        "pocketsphinx",
    ],
)
