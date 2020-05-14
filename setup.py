from setuptools import setup

setup(name='error_correction',
      version='1.0',
      description='Analyze chromosome segregation error data',
      url='http://github.com/phys201/error_correction',
      author='gloriaha',
      author_email='gloriaha@users.noreply.github.com',
      license='GPLv3',
      packages=['error_correction'],
      install_requires=['numpy', 'matplotlib', 'pandas', 'scipy', 'seaborn', 'emcee', 'pyyaml', 'tqdm'],
      zip_safe=False)
