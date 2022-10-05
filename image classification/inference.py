#TODO: Import dependencies for Debugging andd Profiling
#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
#import sagemaker
import argparse
import os 
from torchvision.transforms.transforms import ToTensor
import logging

#import smdebug.pytorch as smd


def test(model, test_loader , cost_function ):#, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    #hook.set_mode(smd.modes.TRAIN)                                                      
    running_loss = 0
    correct = 0
    for data , target in test_loader:
      predicted = model(data)
      running_loss += cost_function(predicted , target)
      predicted = predicted.max(1,keepdim = True)[1]
      correct += predicted.eq(target.view_as(predicted)).sum().item()
    logger.info(f'test data loss is {running_loss} and accuracy is {correct*100/len(test_loader.dataset)}')


    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = torchvision.models.resnet50(pretrained = True)
    for parameter in model.parameters():
        parameter.requires_grad = False

    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(nn.Linear(num_features , 3)) ##//to be edited tp the number of classes you have

    return model

###########################
## this is for inference ##
###########################
def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    #model.to(device).eval()
    return model

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

    

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters() , lr=0.001 )
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
   
    #sagemakersession = sagemaker.Session()
    #bucket = 'dogimages'#'s3://dogimages/dogImages/'#url of the s3 bucket
    #sagemakersession.download_data('data', bucket=bucket)
    train_transform = transforms.Compose(transforms=[transforms.Resize((224,224)),transforms.ToTensor() ,transforms.Normalize(0.5,0.5) ])
    train_dataset = torchvision.datasets.ImageFolder(os.path.join(args.train_dir,'train/') , transform=train_transform ) ##you can use load_dataset
    train_loader = torch.utils.data.DataLoader(train_dataset , batch_size = args.batch_size ,shuffle = True )

    valid_transform = transforms.Compose(transforms=[transforms.Resize((224,224)) , transforms.ToTensor() ,transforms.Normalize(0.5,0.5) ])
    valid_dataset = torchvision.datasets.ImageFolder(args.valid_dir , transform=valid_transform )
    validation_loader = torch.utils.data.DataLoader(valid_dataset , batch_size = args.batch_size ,shuffle = True )

    test_transform = transforms.Compose(transforms=[transforms.Resize((224,224)), transforms.ToTensor() ,transforms.Normalize(0.5,0.5) ])
    test_dataset = torchvision.datasets.ImageFolder(args.test_dir , transform=test_transform )
    test_loader = torch.utils.data.DataLoader(test_dataset , batch_size = args.batch_size ,shuffle = True )
  
    #hook = smd.Hook.create_from_json_file()
    #hook.register_hook(model)
    #hook.register_loss(loss_criterion)

    train(model, train_loader, loss_criterion, optimizer , args.epochs , validation_loader)# , hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion)# , hook)
    
    '''
    TODO: Save the trained model
    '''
    #torch.save(model, args.mode_path)
    save_model(model , args.model_dir)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(
      "--batch-size",
      type=int,
      default=64,
      metavar="N",
      help="input batch size for training (default: 64)",
  )
     
    parser.add_argument(
      "--lr",
      type=float,
      default=0.001,
      metavar="LR",
      help="input learning rate for training (default: 0.001)",
  )
    
    parser.add_argument(
      "--epochs",
      type=int,
      default=2 ,
      metavar="EPOCHS",
      help="number of epochs in the training (default: 2)",
  )
      
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    #this will be s.th like s3://sagemaker-us-east-1-648346130239/sagemaker/DEMO-pytorch-mnist
    parser.add_argument("--valid_dir", type=str , default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])

    args=parser.parse_args()
    
    main(args)
