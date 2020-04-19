from setuptools import setup

setup(name='lf2_gym',
      version='0.1',
      install_requires=['gym',
                        'mss',
                        'pyautogui',
                        'opencv-python>=3.3.1,<4.0.0',
                        'numpy',
                        'pymem',
                        'configobj'])