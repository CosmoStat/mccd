# Python Package Template

---
> Author: <a href="http://www.cosmostat.org/people/sfarrens" target="_blank" style="text-decoration:none; color: #F08080">Samuel Farrens</a>  
> Email: <a href="mailto:samuel.farrens@cea.fr" style="text-decoration:none; color: #F08080">samuel.farrens@cea.fr</a>  
> Year: 2019  
---

This template is designed to enable users to quickly package a Python project. After following the set-up instructions you should be able to run CI tests and automatically generate API documentation for your code.

## Contents
---

1. [Set-Up](#Set-Up)
1. [Management](#Management)
1. [Deployment](#Deployment)
1. [CosmoStat](#CosmoStat)

## Set-Up
---

In order to use this template please follow the instructions provided below.

### Step 1: Get a remote copy of the template

The following instructions are for the GitHub website.

1. Click the `Use this template` button.
2. Specify a `Repository name` for your package. Ideally your package name should match the repository name. Note that you can change the name later if necessary.
3. Click the `Create repository from template` button. This should create a new repository on your GitHub account.
4. Take note of your repository URL.

### Step 2: Configure the template

The following instructions should be run in a terminal on your local machine.

1. Clone your remote repository using your repository URL. *e.g.* for a package called `mypackage` you would run:
```bash
git clone https://github.com/username/mypackage
```
2. Go to the package directory. *e.g.*
```bash
cd mypackage
```
3. Run the configuration script and follow the instructions.
```bash
./configure.sh
```
4. If you chose not to run git commands in the configuration script then manually push your changes to GitHub.
```bash
git add .
git commit -m "updated template"
git push
```

### Step 3: Activate CI tests

The following instructions are for the Travis-CI website.

1. Go to https://travis-ci.org/.
2. Log in using your GitHub account details.
3. Go to your list of GitHub repositories by clicking on the `+` next to `My Repositories` (or go to https://travis-ci.org/account/repositories).
4. Find the package name you provided for the repository and click the switch on the right to activate CI tests.

> Note: if you can't find your repository try clicking the `Sync account` button on the top left and then refresh the page.

> Note: that private GitHub repositories can be accessed via https://travis-ci.com/.

### Step 4: Activate coverage tests

The following instructions are for the Coveralls website.

1. Go to https://coveralls.io/.
2. Log in using your GitHub account details.
3. Go to the menu on the left and click on `Add repos`.
4. Find the package name you provided for the repository and click the switch on the left to activate coverage tests.

> Note: that private GitHub repositories require a pro account on Coveralls.

### Step 5: Automate API documentation build

The objective of this step is to allow Travis to automatically generate your API documentation every time you make a pull request.

1. If you chose not to run git commands in the configuration script then manually create a `gh-pages` branch:
```bash
git checkout -b gh-pages
git rm -rf .
git push --set-upstream origin gh-pages
git commit -m "cleaning gh-pages"
git push
git checkout master
git branch -d gh-pages
```
2. On your GitHub page, go to `Settings>Developer settings>Personal access tokens` (note that these are the user settings in the top right menu bar of GitHub and not the repository settings).
3. Click on the `Generate new token` button, provide a note such as "API Documentation", click the `public_repo` checkbox and finally click the `Generate token` button at the bottom of the page.
4. Copy the token string and be sure to **save the token for future use!** Once you have an active token you can use it for other repositories.
5. On the Travis website, click on your package and go to `More options>Settings`.
6. Under `Environment Variables` provide the name `GH_TOKEN`, then paste the value of your personal access token, and click the `Add` button. Leave the other options as they are.

> More detailed instructions are provided in the [travis-sphinx documentation](https://github.com/Syntaf/travis-sphinx).

### Step 6: Run tests

Please do the following to make sure everything is working as expected.

1. On your machine, create a new branch. *e.g.*
```bash
git checkout -b new_branch
```
2. Make a modification to any of the files. *e.g.* add a new function to `example/math.py`.
```python
def add_float(x, y):
    """Add Floats

    Add two float values.

    Parameters
    ----------
    x : float
        First value
    y : float
        Second value

    Returns
    -------
    float
        Result of addition

    Raises
    ------
    TypeError
        For invalid input types.

    """

    if not isinstance(x, float) or not isinstance(y, float):
        raise TypeError('Inputs must be floats.')

    return x + y
```
3. Add, commit and push your changes.
```bash
git add .
git commit -m "testing"
git push origin new_branch
```
4. Go to the your remote repository and click on the `Compare & pull request` button that should have appeared for your branch.
5. Provide a title and description for your pull request and click the `Create pull request` button.
6. Once open, your pull request should automatically launch the Travis CI tests. Note that this may take a few seconds to start. Click on the link that appears if you want to follow the progress of the tests.
7. Once your CI tests have passed you can merge your pull request, which should automatically generate your package API documentation. Go to *e.g.* https://username.github.io/mypackage/ to view your documentation.

## Management
---

Now that your package is set up you can start managing your own code.

### Add new content

1. Add new modules following the contents of the `example` folder. Be sure to include a `__init__.py` file in every new directory you create.
2. Add new submodules following the contents of `hello.py` and `math.py`. Be sure to follow the [Numpy docstring conventions](https://numpydoc.readthedocs.io/en/latest/format.html) in writing your API documentation.
3. Write unit tests as you add new functions and classes to retain the highest possible code coverage. Follow the examples in `tests`.
4. To manage your code development you should follow the procedure described in Step 6 of the set-up. Note that you can continue to work on a branch you created for an open pull request up until it is merged. After merging a branch it is good practice to delete it.

### Clean up

1. Once you have added some of your own content remove the `example` module and corresponding tests.
2. Update `info.py` with a list of your code dependencies so that they can be installed automatically with your package.
3. Replace this `README.md` with your own package documentation.
4. Update `docs/source/index.rst` with a more detailed description of your package.

### Optional

1. Define your package contribution guidelines in `CONTRIBUTING.md`.
2. Customise your API documentation in `docs/source/conf.py`.
3. Customise your CI and coverage tests in `setup.cfg`.
4. Customise the running of your CI tests in `.travis.yml`.
5. Add badges to your `README.md` from *e.g.* [Travis](https://docs.travis-ci.com/user/status-images/).

### Local Tests

To speed up development of your package it may be convenient to run tests locally before submitting a pull request. This can be done as follows:

1. Install the developer tools.
```bash
pip install -r develop.txt
```
2. Run the tests locally.
```bash
python setup.py test
```


## Deployment
---

In order to upload your package to [PyPi](https://pypi.org/) (which allow users to install your package with `pip`), you should follow these steps:

1. Check if your package name has already been taken. *e.g.*
```bash
pip search mypackage
```
2. If the name is already taken, you may want to rename your package.
3. Create an account on https://pypi.org/.
4. Install twine.
```bash
pip install twine
```
5. Build a distribution of your package.
```bash
python setup.py sdist bdist_wheel
```
6. Finally, upload your distribution to PyPi.
```bash
twine upload dist/*
```
7. Your package can now be installed anywhere using `pip`. *e.g.*
```bash
pip install mypackage
```
8. To update your package, change the version number in `info.py` and repeat steps 5 and 6.

> Note that step 6 can be simplied by creating a [.pypirc](https://docs.python.org/3.3/distutils/packageindex.html#pypirc) file.

## CosmoStat
---

All members of CosmoStat should aim to migrate mature repositories to the [CosmoStat group on GitHub](https://github.com/CosmoStat). This way your code will get more visibility and be maintained by the team.

1. Go to your repository `Settings`.
2. Scroll down to the `Danger Zone`.
3. Click the `Transfer` button.
4. Type the name of your repository and for the new owner write `CosmoStat`.
5. Now you can fork this repository and continue to develop.
