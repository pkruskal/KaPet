KaPet
====

# Summary
https://www.kaggle.com/c/petfinder-adoption-prediction

## submissions and output
Models are submitted as jupyter notebooks so all submissions should be in that form and saved in `/submissions`. 
Packages and code can be uploaded to Kaggles servers and then imported in your notebook.
Idealy the notebook is mostly just calling your classes and functions

## configuration parameters
- see `configuration.py` for documentation on configuration paramaters

# Models

## simple NN backbone followed by fully connected network predictor
- build a simple CNN (or image transformer) to generate a feature array. 
- add on features of the metadata and 

# Dev setup

Download data to the `./data` directory.
https://www.kaggle.com/c/petfinder-adoption-prediction/data

## Option 1: Docker 
Build: (must rebuild anytime dependencies change)
```shell
docker-compose build
```

Run bash environment:
```shell
docker-compose run --rm kapet 
```

## Option 2: local
- Python 3.8 environment
- `pip install -r requirements.txt`
