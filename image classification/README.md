# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This project is done on a reduced dataset of the classical dog breed classication data set.
 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in kaggle: https://www.kaggle.com/datasets/venktesh/person-images

I used a reduced version of the same data set including only 3 classes in the train, validation and test which are (001.Affenpinscher , 002.Afghan_hound , 003.Airedale_terrier)

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
- lr : : ContinuousParameter(0.001, 0.1),
- "batch-size": CategoricalParameter([32, 64, 128, 256, 512]), ## edit to integerparameter

## best hyperparameters:
- lr : 0.003433
- batch size : 64


## Debugging and Profiling
I used the debugger to track the loss function of both the training dataset and validation dataset and plotting them.

### Results
according to the plot out of the debugger:
there is overfit and the test steps are larger as we test the model on the test data after the evaluation data



## Model Deployment
to query the endpoint we just need to resize the image(3,224,224) and convert to tensor and then convert to numpy array and send to the model


