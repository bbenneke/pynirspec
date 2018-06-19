from distutils.core import setup

setup(name='pynirspec',
      version='0.1.0',
      description='Keck-NIRSPEC reduction library',
      author='Klaus Pontoppidan',
      author_email='pontoppi@stsci.edu',
      url='http://www.stsci.edu/~pontoppi',
      packages=['pynirspec','utils'],
      package_data={'pynirspec': ['*.ini']}
      )

    
