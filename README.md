# Model deployment project


Please do not fork this repository, but use this repository as a template for your refactoring project. Make Pull Requests to your own repository even if you work alone and mark the checkboxes with an x, if you are done with a topic in the pull request message.

## Project for today
The task for today you can find in the [project-description.md](project-description.md) file.
# Guide:
* Train and deploy Random Forest Regressor model on the yellow taxi dataset in: [src](src) folder.
* Local deployment with FastAPI in: [fast_api_deployment_local](fast_api_deployment_local) folder.
* Cloud deployment with FastAPI in: [fast_api_deployment_online](fast_api_deployment_online) folder.
# Questions:

#### 1. RMSE of my model:
RMSE of my model is 4.364

#### 2. What would I do differently if you had more time?
* Add more features to the model.
* Use XGBoost or LightGBM instead of RandomForestRegressor.


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

