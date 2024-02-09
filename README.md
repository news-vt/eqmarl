# eqmarl
eQMARL: Quantum multi-agent reinforcement learning with entangled agents.

## Installation

Installation of this repo is a little finicky because of the requirements for `tensorflow-quantum` on various systems. See instructions below for your particular use case.

### Bare-metal

Python package dependencies can be installed via:

```bash
$ pip install -r requirements.txt -r requirements-dev.txt
```

However, if you are using Anaconda then read the section [Anaconda: A special case](#anaconda-a-special-case) for special install details.

#### Anaconda: A special case

If you are using Anaconda to manage Python then be aware that the version of Python may have been built using an outdated version of macOS. To check this, you can run:

```bash
$ python -c "from distutils import util; print(util.get_platform())"
macosx-10.9-x86_64
```

Notice that in the above example we see the installation of Python was built against `macosx-10.9-x86_64`, however, the wheel for `tensorflow-quantum` requires `macosx-12.1-x86_64` or later.

To mitigate this, you can download the wheel for `tensorflow-quantum==0.7.2` from here <https://pypi.org/project/tensorflow-quantum/0.7.2/#files> and change the name of the filename from `tensorflow_quantum-0.7.2-cp39-cp39-macosx_12_1_x86_64.whl` to `tensorflow_quantum-0.7.2-cp39-cp39-macosx_10_9_x86_64.whl`. Once you've done that you can install the wheel via:

```bash
# Activate your environment.
$ conda activate myenv

# Install wheel file manually.
$ pip install tensorflow_quantum-0.7.2-cp39-cp39-macosx_10_9_x86_64.whl
```

## Docker

The repo comes with a Docker configuration via `Dockerfile` (see [Dockerfile](./Dockerfile)) using the `python:3.9` base image.

## VSCode Devcontainer

The repo also comes with a VSCode devcontainer setup (see [devcontainer.json](./.devcontainer/devcontainer.json)) if you want to work within the Docker environment and use VSCode's editing capabilities natively.