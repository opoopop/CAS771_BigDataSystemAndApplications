from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from transformers import ViTForImageClassification,SwinForImageClassification
from model import CNN_0_A,CNN_0_B
import torch
"""
helper functions
"""




def create_filter_list_vit(file_list,device):
    """
    load the filter model from file pathes 
    """
    result=[]
    for f in file_list:
        model = ViTForImageClassification.from_pretrained(f,num_labels=3)
        model.to(device)
        model.eval()
        result.append(model)
    return result
def create_filter_list_swin(file_list,device):
    """
    load the filter model from file pathes 
    """
    result=[]
    for f in file_list:
        model = SwinForImageClassification.from_pretrained(f,num_labels=3)
        model.to(device)
        model.eval()
        result.append(model)
    return result
def create_classifier_list(file_list,device,task):
    result=[]
    # load classifiers
    for f in file_list:
        if task=='A':
            model =CNN_0_A() 
        else:
            model=CNN_0_B() 
        model.load_state_dict(torch.load(f))
        model.to(device)
        model.eval()
        result.append(model)
    return result
