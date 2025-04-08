import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from meta_model import pipeline_meta
import random
import matplotlib.pyplot as plt
"""
merged model pipeline
"""
upper_map={
0:10,
2:5,
6:11
}

def classifier_result(model,input_tensor,classifier_num,task):
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        _, predicted_class = torch.max(probabilities, dim=1)
        result=classifier_num*5+predicted_class.item()
        if task=='B':
            if result in upper_map:
                #print(result)
                result=upper_map[result]# overlap index
                #print(result)
        return result
def merged_model_A(input_tensor, classifier_list, filter_model_list, device,meta_path,cond_threshold=0.6):
    """
    geive the prediction of input image using 3 classifiers and a 
    model to decide which classifier to chose
    """
    filter_tensor = torchvision.transforms.functional.resize(input_tensor, (224, 224))
    
    #print(filter_tensor)
    pre_set=set()
    result=[]
    n=len(filter_model_list)
    for model in filter_model_list:
        with torch.no_grad():
            output = model(filter_tensor)
            logits = output.logits if hasattr(output, "logits") else output
            probabilities = F.softmax(logits, dim=1)
            max_prob, predicted_class = torch.max(probabilities, dim=1)
            max_prob = max_prob.item()
            predicted_class=predicted_class.item()
            pre_set.add(predicted_class)
            result.append([predicted_class,max_prob])
    
    confidence_score=0
    for x in result:
        confidence_score+=x[1]
    confidence_score/=n # weight 1/n
    #print(confidence_score)
    #print(cond_threshold)
    if len(pre_set)!=1 or confidence_score<=cond_threshold:
        #return -1
        #return pieline2(input_tensor,classifier_list,device)
        return pipeline_meta(input_tensor,classifier_list,device,meta_path,'A')
    else:
        #return -1
        return classifier_result(classifier_list[result[0][0]],input_tensor,result[0][0],'A')

def merged_model_B(input_tensor, classifier_list, filter_model_list,device,meta_path,cond_threshold=0.6):
    """
    geive the prediction of input image using 3 classifiers and a 
    model to decide which classifier to chose
    """
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    op=input_tensor.float()/255.0
    nor_inputs = normalize(op)
    filter_tensor = torchvision.transforms.functional.resize(nor_inputs, (224, 224))
    
    #print(filter_tensor)
    pre_set=set()
    result=[]
    n=len(filter_model_list)
    for model in filter_model_list:
        with torch.no_grad():
            output = model(filter_tensor)
            logits = output.logits if hasattr(output, "logits") else output
            probabilities = F.softmax(logits, dim=1)
            max_prob, predicted_class = torch.max(probabilities, dim=1)
            max_prob = max_prob.item()
            predicted_class=predicted_class.item()
            pre_set.add(predicted_class)
            result.append([predicted_class,max_prob])
    
    confidence_score=0
    for x in result:
        confidence_score+=x[1]
    confidence_score/=n # weight 1/n
    #cond_threshold=0.6
    #print(confidence_score)
    if len(pre_set)!=1 or confidence_score<=cond_threshold:
        #return -1
        #return pieline2(input_tensor,classifier_list,device)
        return pipeline_meta(input_tensor,classifier_list,device,meta_path,'B')
    else:
        #return -1
        return classifier_result(classifier_list[result[0][0]],input_tensor,result[0][0],'B')

def test_merged_model(test_set,classifier_list,filter_list,device,threshold,meta_path,task):
    """
    test the result of merged model
    """
    total=0
    correct=0
    with torch.no_grad():
        for i in range(len(test_set)):
            if i%100==0:
                print(f'testing on number {i}')
            image, label = test_set[i]  
            image = image.unsqueeze(0).to(device)  # 
            #label = torch.tensor([label], device=device)
            if task=='A':
                output= merged_model_A(image,classifier_list,filter_list,device,meta_path,cond_threshold=threshold)
            else:
                output= merged_model_B(image,classifier_list,filter_list,device,meta_path,cond_threshold=threshold)
            #_, predicted = torch.max(output, 1)
            if output==-1:
                continue
            total += 1
            correct += (output == label)
            #if output in [10,5,11]:
                #print(output == label)


    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    #print(total)


op_map_B={
    173: 'Chihuahua',
    137: 'baboon',
    34: 'hyena',
    159: 'Arctic_fox',
    201: 'lynx',
    202: 'African_hunting_dog',
    80: 'zebra',
    135: 'patas',
    24: 'African_elephant',
    130: 'boxer',
    124: 'collie',
    125: 'golden_retriever'
}

rever_map_B={
    10: 173, 
    1: 137, 
    5: 34, 
    3: 159, 
    4: 201, 
    11: 202, 
    7: 80, 
    8: 135, 
    9: 24, 
    12: 130, 
    13: 124, 
    14: 125
}

def display_sample_B(test_set, classifier_list, filter_list, device, meta_path,threshold,display_num=10):
    random.seed(72)
    rd_list = [i for i in range(len(test_set))]
    random.shuffle(rd_list)
    sample_list = rd_list[:display_num]
    print(f'index of the sample displayed {sample_list}')
    
    display_samples = []
    for i in sample_list:
        image, label = test_set[i]
        image = image.unsqueeze(0).to(device)  
        output = merged_model_A(image,classifier_list,filter_list,device,meta_path,cond_threshold=threshold)
        display_samples.append((image.squeeze().cpu(), op_map_B[rever_map_B[label.item()]], op_map_B[rever_map_B[output]]))
    
    print(f'Displaying {display_num} samples: ')
    for i, (img, true_label, pred_label) in enumerate(display_samples):
        plt.subplot(2, 5, i + 1)  
        

        img = img.permute(1, 2, 0) 

        img = img.to(torch.uint8) 
        
        plt.imshow(img.numpy()) 
        plt.title(f"True: {true_label} \n  Pred: {pred_label}", fontsize=7)
        plt.axis('off') 

    plt.subplots_adjust(wspace=1.3, hspace=0.1)
    plt.show()


op_map_A={
    0: 'apple',
    10: 'bowl',
    20: 'chair',
    30: 'dolphin',
    40: 'lamp',
    1: 'aquarium_fish',
    11: 'boy',
    21: 'chimpanzee',
    31: 'elephant',
    41: 'lawn_mower',
    2: 'baby',
    12: 'bridge',
    22: 'clock',
    32: 'flatfish',
    42: 'leopard'
}
#{0: 0, 10: 1, 20: 2, 30: 3, 40: 4, 1: 5, 11: 6, 21: 7, 31: 8, 41: 9, 2: 10, 12: 11, 22: 12, 32: 13, 42: 14}
rever_map_A={
    0: 0,
    1: 10,
    2: 20,
    3: 30,
    4: 40,
    5: 1,
    6: 11,
    7: 21,
    8: 31,
    9: 41,
    10: 2,
    11: 12,
    12: 22,
    13: 32,
    14: 42
}
def display_sample_A(test_set, classifier_list, filter_list, device, meta_path,threshold,display_num=10):
    random.seed(42)
    rd_list = [i for i in range(len(test_set))]
    random.shuffle(rd_list)
    sample_list = rd_list[:display_num]
    print(f'index of the sample displayed {sample_list}')
    
    display_samples = []
    for i in sample_list:
        image, label = test_set[i]
        image = image.unsqueeze(0).to(device)  
        output = merged_model_B(image,classifier_list,filter_list,device,meta_path,cond_threshold=threshold)
        display_samples.append((image.squeeze().cpu(), op_map_A[rever_map_A[label]], op_map_A[rever_map_A[output]]))
    
    print(f'Displaying {display_num} samples: ')
    for i, (img, true_label, pred_label) in enumerate(display_samples):
        plt.subplot(2, 5, i + 1)  
        

        img = img.permute(1, 2, 0) 
        img = img / 2 + 0.5

        plt.imshow(img.numpy()) 
        plt.title(f"True: {true_label} \n  Pred: {pred_label}", fontsize=7)
        plt.axis('off') 

    plt.subplots_adjust(wspace=1.3, hspace=0.1)
    plt.show()
