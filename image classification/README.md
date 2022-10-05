# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.
I used a reduced version of the same data set including only 3 classes in the train, validation and test

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search
- lr : : ContinuousParameter(0.001, 0.1),
- "batch-size": CategoricalParameter([32, 64, 128, 256, 512]), ## edit to integerparameter

## best hyperparameters:
- lr : 0.003433
- batch size : 64

## the completed training job:
image.png

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?
there is overfit and the test steps are larger as we test the model on the test data after the evaluation data

**TODO** Remember to provide the profiler html/pdf file in your submission.
It's provided in the train_and_deploy.ipynb


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
to query the endpoint we just need to resize the image and convert to tensor and then convert to numpy array and send to the model

unfortunately I removed the end point because I was about to run out of the cost

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
