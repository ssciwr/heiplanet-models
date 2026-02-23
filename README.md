# heiplanet-models
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ssciwr/heiplanet-models/main.svg)](https://results.pre-commit.ci/latest/github/ssciwr/heiplanet-models/main)
![License: MIT](https://img.shields.io/github/license/ssciwr/heiplanet-models)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/heiplanet-models/ci.yml?branch=main)
[![codecov](https://codecov.io/gh/ssciwr/heiplanet-models/graph/badge.svg?token=ZIgvUHnpkD)](https://codecov.io/gh/ssciwr/heiplanet-models)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_onehealth-model-backend&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=ssciwr_onehealth-model-backend)
![Language](https://img.shields.io/github/languages/top/ssciwr/heiplanet-models)
[![Documentation](https://img.shields.io/badge/Documentation-online-brightgreen)](https://ssciwr.github.io/heiplanet-models/) [![License](https://img.shields.io/github/license/ssciwr/heiplanet-models)](https://github.com/ssciwr/heiplanet-models/blob/main/LICENSE)

## Description
This repository consists of a collection of models that share the same infrastructure code, used for the [heiplanet project](http://129.206.4.157/)


## Installation
## For usage
It is strongly recommended to use a virtual environment to install packages into. This will keep the package self-contained without its dependencies polluting the system, e.g. with python venv:

```bash
python3 -m venv .venv # creates a venv name 'venv' in a hidden directory on unix

source ./.venv/bin/activate # activate the environment
```

Then, install the package from pypi:
```bash
pip install heiplanet-models
```
In order to be able to visualize the computational graphs of the models you use or build, you need to install the package with the `viz` option. This will install the graphviz python package which will take care of the visualization.

```bash
pip install heiplanet-models[viz]
```
Note that if you are using zsh (default on macos), you need to add quotes

```bash
pip install "heiplanet-models[viz]"
```
Graphviz itself has additional dependencies it needs to install. For more details, see [here](https://github.com/xflr6/graphviz?tab=readme-ov-file#installation).

## For development
Clone the repository

```bash
git clone https://github.com/ssciwr/heiplanet-models.git
```
Create some virtual environment as described above.
Then, go to the base directory of the repository, and run
```bash
pip install -e .
```
or including the graphviz dependency:

```bash
pip install -e .[viz]
```
again, you need to take care of shell specifics, e.g. on zsh:

```bash
pip install -e ".[viz]"
```
Please note the remark on graphviz's dependencies above.

## Troubleshooting
to be done.
