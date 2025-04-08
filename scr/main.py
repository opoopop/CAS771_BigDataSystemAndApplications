from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from data_processing import load_test_data_A,load_test_data_B
from utils import create_classifier_list,create_filter_list_swin,create_filter_list_vit
from merged_model import test_merged_model,display_sample_B,display_sample_A
import torch
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="experiment args")
    parser.add_argument("--task", type=str, default='A', help="chose to test on A or B")
    #parser.add_argument("--threshold", type=int, default=0.6, help="threshold of the filter")
    #parser.add_argument("--threshold_set", type=bool, default=False, help="change into own threshold")
    args = parser.parse_args()
    threshold=0.6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task=='A':
        threshold=0.6
        test_set,test_loader=load_test_data_A()
        filter_file_list=[
        "model/taskA/gating_network/filter_model_1/"
        ]
        classifier_file_list=[
        'model/taskA/model1/model_weights_CNN_0.pth',
        'model/taskA/model2/model_weights_CNN_0.pth',
        'model/taskA/model3/model_weights_CNN_0.pth'

        ]
        meta_path="model/taskA/meta_model/meta_model_1.pth"
    else:
        threshold=0.6
        test_set,test_loader=load_test_data_B()
        filter_file_list=[
        "model/taskB/gating_network/filter_model_1/",
        "model/taskB/gating_network/filter_model_2/",
        "model/taskB/gating_network/filter_model_3/"
        ]
        classifier_file_list=[
        'model/taskB/model1/model_weights_CNN_0.pth',
        'model/taskB/model2/model_weights_CNN_0.pth',
        'model/taskB/model3/model_weights_CNN_0.pth'

        ]
        meta_path="model/taskB/meta_model/meta_model_1.pth"

    if args.task=='A':
        filter_list=create_filter_list_swin(filter_file_list,device)
    else:
        filter_list=create_filter_list_vit(filter_file_list,device)

    
    classifier_list=create_classifier_list(classifier_file_list,device,args.task)
    test_merged_model(test_set,classifier_list,filter_list,device,threshold,meta_path,args.task)
    if args.task=='B':
        display_sample_B(test_set, classifier_list, filter_list, device,  meta_path,threshold,display_num=10)
    else:
        display_sample_A(test_set, classifier_list, filter_list, device,  meta_path,threshold,display_num=10)
