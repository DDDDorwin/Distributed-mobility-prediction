![Tests](https://github.com/Mithzyl/Project_CS_UserVsSpecific/actions/workflows/tests.yml/badge.svg)

# Project_CS_UserVsSpecific
Project CS  Group User. Vs Specific


### Project Setup
Install pdm package manager with ``pip install pdm``

Clone the project and run ``pdm install`` to install project dependencies and make the project be installed in
editable mode.

### Adding new packages
To add a new package or dependency run ``pdm add <package>``

### Running the project
Pull the project and setup the project as described above.
Then install the project as a package to your python environment

``$ pip install pdm``

**Note:** If you change something in the files you will need to run ``$ pdm sync`` before running the above command.

This will make the package available to your python environment. And this package will be editable. Anytime you edit
your modules you will need to run the above command again to update your python environment with the changes.

To run the project you can now simply run

```$ gvslearning```

In your terminal. This will run the main entry point file in the gvslearning package. Which is located in the
``src/main/__main__.py`` file.

### Running the tests
To run the in temporary virtual environments you can run in terminal

``$ tox``

in the project root directory. This will run all the tests in the
``tests`` directory. Hence, all tests should be put in this directory. The tests will also include test-coverage.
And shows linting issues too

## Example of editing the project
To edit the project you can simply open the project in your favourite IDE. And start editing the files.
For example, if you want to edit the ``src/main/__main__.py`` file. You can simply open the file and edit it.
Then run the command ``$ pipx install -e .`` again to update your python environment with the changes.
Then you can run the project again with ``$ gvslearning``

To import other modules in the project you can simply use the ``from gvslearning import <module>`` syntax.
Each file in the ```src/``` is a module. So if you want to import the ``src/model/modeled.py`` file in the
``src/main/__main__.py`` directory you can simply use the ``from model.modeled import example_model`` syntax.
**See example in the ``src/main/__main__.py`` file**
