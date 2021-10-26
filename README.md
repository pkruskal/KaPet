KaPet
====

# Summary
https://www.kaggle.com/c/petfinder-adoption-prediction

## Submissions and output
Models are submitted as jupyter notebooks so all submissions should be in that form and saved in `/submissions`. 
Packages and code can be uploaded to Kaggles servers and then imported in your notebook.
Idealy the notebook is mostly just calling your classes and functions

## Configuration parameters
See `configuration.py` for documentation on configuration paramaters

# Models

## Simple NN backbone followed by fully connected network predictor
- build a simple CNN (or image transformer) to generate a feature array. 
- add on features of the metadata and 

# Dev setup
## Data
Download data to the `./data` directory.
https://www.kaggle.com/c/petfinder-adoption-prediction/data

## Option 1: Docker environment
 Build: (must rebuild anytime dependencies change)
```shell
docker-compose build
```

Run jupyter lab:
```shell
docker-compose up 
```
then follow the link in your local browser (`http://127.0.0.1:8888/lab?token=xxxx`) 

When done, to clean-up:
```shell
docker-compose down
```

Alternatively, to run a simple bash environment in the container:
```shell
docker-compose run --rm --entrypoint bash kapet  
```

## Option 2: local environment
- Python 3.8 environment
- `pip install -r requirements.txt`
