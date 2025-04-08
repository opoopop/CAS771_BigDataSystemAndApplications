import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import  TensorDataset
"""
preprocess and load the data
"""
def load_test_data_A():
    """
    return the task A test data
    """
    all_class = [[0, 10, 20, 30, 40],
                [1, 11, 21, 31, 41],
                [2, 12, 22, 32, 42]]


    selected_classes=sum(all_class, [])
    label_mapping = {}
    label_idx = 0
    for st in all_class:
        for orig_label in st:
            label_mapping[orig_label] = label_idx
            label_idx += 1


    #print(label_mapping)


    cifar100_classes = torchvision.datasets.CIFAR100(root='./data', download=False).classes


    transform_test = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])



    test_set = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=False, transform=transform_test
    )

    selected_indices_test = [
        idx for idx, (_, label) in enumerate(test_set)
        if label in selected_classes
    ]
    #selected_indices_test=selected_indices_test[:100]

    for i in selected_indices_test:
        test_set.targets[i]=label_mapping[test_set.targets[i]]


    filtered_test_set = Subset(test_set, selected_indices_test)
    test_loader = DataLoader(filtered_test_set, batch_size=32, shuffle=False, num_workers=2)
    print('Finish data loading on Task A')
    return filtered_test_set,test_loader

def load_data(file_path):
    raw_data = torch.load(file_path)
    data = raw_data['data']  
    labels = raw_data['labels']   
    if data.ndim == 4 and data.shape[-1] == 3:  
        data = data.permute(0, 3, 1, 2) 
    mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels], dtype=torch.long)

    return TensorDataset(data, mapped_labels)

def load_mutifile_data(file_paths,label_mapping):
    all_data = []
    all_labels = []
    
    for file_path in file_paths:
        raw_data = torch.load(file_path)
        data = raw_data['data'] 
        labels = raw_data['labels'] 
        
        if data.ndim == 4 and data.shape[-1] == 3:  
            data = data.permute(0, 3, 1, 2) 
        #data = data.float() / 255.0
        #normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        #normalized_data = normalize(data)
        mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels], dtype=torch.long)
        
        all_data.append(data)
        all_labels.append(mapped_labels)
    
    # 拼接所有数据和标签
    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return TensorDataset(all_data, all_labels)
def load_test_data_B():

    all_class=[[173, 137, 34, 159, 201],
            [34, 202, 80, 135, 24],
            [173, 202, 130, 124, 125]]

    selected_classes=sum(all_class, [])
    label_mapping = {}
    label_idx = 0
    for st in all_class:
        for orig_label in st:
            label_mapping[orig_label] = label_idx
            label_idx += 1
    #print(label_mapping)

    bt_size=32

    test_file_paths = [
        'data/Task2_data/val_dataB_model_1.pth',
        'data/Task2_data/val_dataB_model_2.pth',
        'data/Task2_data/val_dataB_model_3.pth'
    ]
    test_set=load_mutifile_data(test_file_paths,label_mapping)


    test_loader = DataLoader(test_set, batch_size=bt_size, shuffle=False, num_workers=4)

    print('Finish data loading on Task B')
    #print(f"Training data size: {len(train_set)}")
    #print(f"Testing data size: {len(test_set)}")

    return test_set,test_loader

