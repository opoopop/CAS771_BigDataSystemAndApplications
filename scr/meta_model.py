import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from model import MetaModel
"""
meta model pipeline for merged model
"""
upper_map={
0:10,
2:5,
6:11
}

def get_meta_data(input,classifier_list):
    """
    create the data as the input of meta model
    """
    with torch.no_grad():
        model_outputs = []
        for model in classifier_list:
                output = model(input)
                model_outputs.append(output)

        combined_outputs = torch.cat(model_outputs, dim=1)
    return combined_outputs


def pipeline_meta(input_tensor, classifier_list,device,meta_path,task):
    model=MetaModel()
    #model=torch.load('/home/chunjielu/CIFAR100classifier/meta/meta_model_1.pth') 
    model.load_state_dict(torch.load(meta_path, weights_only=False))
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(get_meta_data(input_tensor,classifier_list))
        val, predicted = torch.max(outputs, dim=1)
    #if val<0.9:
        #return pieline2(input_tensor, classifier_list,device)
    result=predicted.item()
    if task=='B' and result in upper_map: # overlap index in task B
        result=upper_map[result]
    return result