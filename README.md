# Model deployment project


Please do not fork this repository, but use this repository as a template for your refactoring project. Make Pull Requests to your own repository even if you work alone and mark the checkboxes with an x, if you are done with a topic in the pull request message.

## Project for today
The task for today you can find in the [project-description.md](project-description.md) file.




## Setup
### Pipenv
```bash
pyenv local 3.10.9
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Poetry
```shell
poetry config virtualenvs.in-project true
poetry install
```


```shell
source .venv/bin/activate
poetry add $( cat requirements.txt ) 	
```

