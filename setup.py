from setuptools import setup
from setuptools.command.install import install

setup(
    name='PASE',
    version='0.1.1-dev',
    packages=['pase', 'pase.models', 'pase.models.WorkerScheduler', 'pase.models.Minions'],
    cmdclass={'install': install},
)