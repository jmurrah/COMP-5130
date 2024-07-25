# How to run
1) Update environemt
```
sudo apt update
```

2) Install pipx: https://pipx.pypa.io/stable/
```
sudo apt install pipx

pipx --version
```

3) Download poetry: https://python-poetry.org/docs/#installing-with-pipx
```
pipx install poetry==1.8.2

poetry --version
```

4) Check python version. If your python version is >= 3.9.1 and < 3.11 continue to step 7. Otherwise you must switch to a python version that is >= 3.9.1 and < 3.11
```
python --version
```

5) Download pyenv: https://github.com/pyenv/pyenv?tab=readme-ov-file#installation
```
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

curl https://pyenv.run/ | bash

echo -e 'export PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc

echo -e 'eval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc

exec "$SHELL"

pyenv --version
```

6) Download and switch to python 3.10.0
```
pyenv install 3.10.0

pyenv global 3.10.0

python --version
```

7) Launch poetry shell
```
poetry shell
```

8) Download dependencies
```
poetry install
```

9) Run k-means algorithm
```
poetry run python3 k_means.py
```

# Presentation
Watch the K-Means project presentation [here](https://www.youtube.com/watch?v=Cop9CxSUmLo).
