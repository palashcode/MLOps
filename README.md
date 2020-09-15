# MLOps

- MLOps  Consists of ML system development (Dev) and ML system operation(Ops).
- ML ops does automation and monitoring for machine learning models.
- It involves construction, integration, testing, releasing, deployment and infra management of ML system.

### Difference b/w MLOps and DevOps
- Apart from unit and integration testing , data validation, model quality evaluation and model validation is required.
- Even after deployment model needs to be trained continuously.
- Model performance in production needs to be monitored because data profile can change overtime. e.g fashion trends data.

### Steps involved in MLOps
- Data extraction : extract data from various sources
- Data Analysis: explore data analysis, understand data characterstics, prepare data , feature engineering
- Data preparation: clean, split data into train,test and validation set.
- Model training: tuning model
- Model evaluation: evaluate on test set
- Model validation: check predictive performance
- Model serving : deploy model to target env
- Model monitoring: monitor the model performance to replace model


## Instruction on running the script
Create virtual environment using ```pipenv shell```
To create new model run ``` python train_pipeline.py ```
Models are saved in ```models``` folder file name contains timestamp.
File paths are defined in ```config.py```
data files are stored in ```data``` folder.

To run validation/prediction on new dataset update ```VALIDATION_DATA_FILE``` in  ```config.py```.
Put the data file in data folder. 
Then run ```python predict.py```.


## Task Not done
- model validation
- model performance monitoring
- deployment to CI/CD plateform for automating the process.
