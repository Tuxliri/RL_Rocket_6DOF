from setuptools import setup

setup(
    name='my_environment',
    version='0.0.2',
    install_requires=['gym==0.21.0', 'scipy==1.7.3','pygame==2.1.2', 'numpy==1.21.*', 'wandb','pyvista','stable_baselines3'],
    #include_package_data=True,
    #package_dir={"": "my_environment"}
    )
