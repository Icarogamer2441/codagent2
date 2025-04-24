import os
from setuptools import setup, find_packages

setup(
    name='codagent2',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'google-generativeai>=0.3.0', # Or the latest version
        'langchain>=0.0.350', # Or the latest version
        'langchain-google-genai>=0.0.1', # Or the latest version
        'rich>=13.0.0', # Or the latest version for beautiful terminal output
        'patch>=1.0', # Add the patch library
        'prompt_toolkit>=3.0.0', # Re-add prompt_toolkit for arrow key support
        'unidiff>=0.7.0', # Add the unidiff library
        # Removed python-dotenv as it's not needed for reading user env vars
        # 'python-dotenv>=1.0.0',
        # Add other dependencies as needed later
    ],
    entry_points={
        'console_scripts': [
            'coda2=codagent2.cli:main',
        ],
    },
    author='Your Name', # Replace with your name
    description='An AI agent for code modifications and terminal tasks',
    long_description=open('README.md').read() if os.path.exists('README.md') else '', # Add a README.md later
    long_description_content_type='text/markdown',
    url='http://github.com/yourusername/codagent2', # Replace with your repo URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Or your preferred license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
