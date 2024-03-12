from setuptools import setup

setup(name='chromatinference',
      version='1.0',
      description='Analyze chromosome segregation error data',
      url='http://github.com/gloriaha/chromatinference',
      author='gloriaha',
      author_email='gloriaha@users.noreply.github.com',
      license='GPLv3',
      packages=['chromatinference'],
      install_requires=['numpy', 'matplotlib', 'pandas', 'scipy', 'seaborn', 'emcee', 'pyyaml', 'tqdm'],
      zip_safe=False)
