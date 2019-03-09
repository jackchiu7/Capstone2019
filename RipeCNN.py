import argparse
import torch
import torch.nn as nn
import torchnet as tnt
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import tensorflow as tf

from sklearn.metrics import confusion_matrix
import numpy as np

from sklearn.utils import compute_class_weight

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='Banana_New/Train')
parser.add_argument('--val_dir', default='Banana_New/Val')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=10, type=int)
parser.add_argument('--num_epochs2', default=10, type=int)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# def imshow(inp, title=None):
    # """Imshow for Tensor."""
    # inp = inp.numpy().transpose((1, 2, 0))
    # inp = IMAGENET_STD * inp + IMAGENET_MEAN
    # inp = np.clip(inp, 0, 1)
    # plt.imshow(inp)
    # if title is not None:
        # plt.title(title)
    # plt.pause(0.001) 
    
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def run_epoch(model, loss_fn, loader, optimizer, dtype):
  """
  Train the model for one epoch.
  """
  # Set the model to training mode
  model.train()
  for x, y in loader:
    # The DataLoader produces Torch Tensors, so we need to cast them to the
    # correct datatype and wrap them in Variables.
    #
    # Note that the labels should be a torch.LongTensor on CPU and a
    # torch.cuda.LongTensor on GPU; to accomplish this we first cast to dtype
    # (either torch.FloatTensor or torch.cuda.FloatTensor) and then cast to
    # long; this ensures that y has the correct type in both cases.
    x_var = Variable(x.type(dtype), requires_grad=True)
    y_var = Variable(y.type(dtype).long())

    # Run the model forward to compute scores and loss.
    scores = model(x_var)
    loss = loss_fn(scores, y_var)

    # Run the model backward and take a step using the optimizer.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

  
def conf_mat(model, loader, dtype):
  conf_matrix = tnt.meter.ConfusionMeter(3)
  for ii, data in enumerate(loader):
    input, label = data
    val_input = Variable(input.type(dtype))
    val_label = Variable(label.type(dtype).long())
    score = model(val_input)
    conf_matrix.add(score.data.squeeze(), label.type(dtype).long())
  
  # tn = float(conf_matrix.conf[0,0])
  # fp = float(conf_matrix.conf[0,1])
  # tp = float(conf_matrix.conf[1,1])
  # fn = float(conf_matrix.conf[1,0])
  # accuracy = (tp+tn)/(tp+tn+fp+fn)
  # precision = tp/(tp+fp)
  # recall = tp/(tp+fn)
  # fscore = (2*precision*recall)/(precision+recall)
  return conf_matrix.conf#, accuracy, precision, recall, fscore
  
def check_accuracy(model, loader, dtype):
  """
  Check the accuracy of the model.
  """
  # Set the model to eval mode
  model.eval()
  num_correct, num_samples = 0, 0
  for x, y in loader:
    # Cast the image data to the correct type and wrap it in a Variable. At
    # test-time when we do not need to compute gradients, marking the Variable
    # as volatile can reduce memory usage and slightly improve speed.
    x_var = Variable(x.type(dtype), requires_grad=False)

    # Run the model forward, and compare the argmax score with the ground-truth
    # category.
    scores = model(x_var)
    _, preds = scores.data.cpu().max(1)
    num_correct += (preds == y).sum()
    num_samples += x.size(0)

  # Return the fraction of datapoints that were correctly classified.
  acc = float(num_correct) / num_samples
  return acc


def main(args):

  dtype = torch.FloatTensor
  classes = ('Underripe', 'Ripe', 'Overripe')

  # Training Data
  train_transform = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),            
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])

  train_dset = ImageFolder(args.train_dir, transform=train_transform)
  train_loader = DataLoader(train_dset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True)

  # Validation Data
  val_transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
  
  val_dset = ImageFolder(args.val_dir, transform=val_transform)
  val_loader = DataLoader(val_dset,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)

  # model = torchvision.models.vgg19(pretrained=True)
  # num_classes = len(train_dset.classes)
  
  # #num_ftrs = model.classifier.in_features
  # #model.classifier = nn.Linear(num_ftrs, num_classes)

  
  # # model.fc = nn.Linear(model.fc.in_features, num_classes)
  
  # # Number of filters in the bottleneck layer
  # num_ftrs = model.classifier[6].in_features
  # # convert all the layers to list and remove the last one
  # features = list(model.classifier.children())[:-1]
  # # # Add the last layer based on the num of classes in our dataset
  # features.extend([nn.Linear(num_ftrs, num_classes)])
  # # # convert it into container and add it to our model class.
  # model.classifier = nn.Sequential(*features)

  # model.type(dtype)
  # loss_fn = nn.CrossEntropyLoss().type(dtype)

  # for param in model.parameters():
    # param.requires_grad = False
  # # for param in model.fc.parameters():
    # # param.requires_grad = True

  # #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  # optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
  
  model = torchvision.models.resnet18(pretrained=True)

  num_classes = len(train_dset.classes)
  model.fc = nn.Linear(model.fc.in_features, num_classes)

  model.type(dtype)
  loss_fn = nn.CrossEntropyLoss().type(dtype)

  for param in model.parameters():
    param.requires_grad = False
  for param in model.fc.parameters():
    param.requires_grad = True

  optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
  
  my_loss = torch.zeros([10,1])
  train_acc = np.zeros([10,1])
  val_acc = np.zeros([10, 1])
  
  for epoch in range(args.num_epochs1):
    # Run an epoch over the training data.
    print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
    my_loss[epoch] = run_epoch(model, loss_fn, train_loader, optimizer, dtype)
    print('Loss: ', my_loss[epoch][0])
    
    print("***********TRAIN METRICS**************\n")
    conf = conf_mat(model, train_loader, dtype)
    print('Confusion matrix: \n', conf)
    train_acc[epoch] = check_accuracy(model, train_loader, dtype)
    print("Accuracy: ", train_acc[epoch])
    # print("Precision: ", precision)
    # print("Recall: ",recall)
    # print("F-score: ",fscore)
    
    print("***********VALIDATION METRICS**************\n")
    conf = conf_mat(model, val_loader, dtype)
    print('Confusion matrix: \n', conf)
    val_acc[epoch] = check_accuracy(model, val_loader, dtype)
    print("Accuracy: ", val_acc[epoch])
    # print("Precision: ", precision)
    # print("Recall: ",recall)
    # print("F-score: ",fscore)
    
    
  # inputs, classes = next(iter(val_loader))
  # # Make a grid from batch
  # out = torchvision.utils.make_grid(inputs)
  # imshow(out)
  
  # get some random training images
  dataiter = iter(val_loader)
  images, labels = dataiter.next()

  # show images
  imshow(torchvision.utils.make_grid(images))
  
  # model.eval()
  # for x, y in val_loader:
    # x_var = Variable(x.type(dtype))
    # scores = model(x_var)
    
  # _, predicted = torch.max(scores, 1)

  # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(17)))
    
  loss = my_loss.detach().numpy()
  plt.plot([1,2,3,4,5,6,7,8,9,10], loss.flatten(), 'g')
  plt.title("Loss Plot")
  plt.xlabel("Loss")
  plt.ylabel("Epoch")
  plt.show()
  
  plt.plot([1,2,3,4,5,6,7,8,9,10], val_acc, 'r', label="validation")
  plt.plot([1,2,3,4,5,6,7,8,9,10], train_acc, 'b', label="train")
  plt.title("Accuracy Plot")
  plt.xlabel("Accuracy")
  plt.ylabel("Epoch")
  plt.legend()
  plt.show()

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)