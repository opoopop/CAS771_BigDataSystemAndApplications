# CAS771_BigDataSystemAndApplications

This is the project of CAS 771:Introduction to Big Data Systems and Applications.

Download the file of **data** and **model** from https://drive.google.com/drive/folders/1OkRybAjtIrF07nQTSVjpl38aMpwEG52t?usp=sharing, then place it in the **root** directory. 

Midterm demo about train lightweight classifier(**this project does not require** ) : https://drive.google.com/drive/folders/1PG8TpA5mRWyIiEwLP8vLOyxLAhP94L6f?usp=drive_link

For your quick review there are some result on notebooks:

1. [CAS771_BigDataSystemAndApplications/notebooks/taskA_merged_model.ipynb at main · opoopop/CAS771_BigDataSystemAndApplications](https://github.com/opoopop/CAS771_BigDataSystemAndApplications/blob/main/notebooks/taskA_merged_model.ipynb) is the result for the merged model of taskA
2. [CAS771_BigDataSystemAndApplications/notebooks/taskB_merged_model.ipynb at main · opoopop/CAS771_BigDataSystemAndApplications](https://github.com/opoopop/CAS771_BigDataSystemAndApplications/blob/main/notebooks/taskB_merged_model.ipynb) is the result for the merged model of taskB
3. [CAS771_BigDataSystemAndApplications/notebooks/gating_network_train.ipynb at main · opoopop/CAS771_BigDataSystemAndApplications](https://github.com/opoopop/CAS771_BigDataSystemAndApplications/blob/main/notebooks/gating_network_train.ipynb) includes the example to train gating network on task A.
4. [CAS771_BigDataSystemAndApplications/notebooks/meta_model_train.ipynb at main · opoopop/CAS771_BigDataSystemAndApplications](https://github.com/opoopop/CAS771_BigDataSystemAndApplications/blob/main/notebooks/meta_model_train.ipynb) includes the example to train meta model on task A.



steps to run the code and get the result.

1. environment set up
​	create virtual environment and install the requirements

```bash
conda create -n myenv
conda activate myenv
```

```bash
cd path/to/your/file
pip install -r requirements.txt
```






2. test of task A

```bash
python scr/main.py --task A
```

result:

```bash
Finish data loading on Task A
testing on number 0
testing on number 100
testing on number 200
testing on number 300
testing on number 400
testing on number 500
testing on number 600
testing on number 700
testing on number 800
testing on number 900
testing on number 1000
testing on number 1100
testing on number 1200
testing on number 1300
testing on number 1400
Test Accuracy: 90.60%
index of the sample displayed [1303, 397, 386, 549, 597, 1113, 1365, 134, 1117, 1164]
Displaying 10 samples:

```

![image-20250408161947244](imgs/taskA.png)

3. test of task B

```bash
python scr/main.py --task B
```

result:

```bash
Finish data loading on Task B
testing on number 0
testing on number 100
testing on number 200
testing on number 300
testing on number 400
testing on number 500
testing on number 600
testing on number 700
Test Accuracy: 78.13%
index of the sample displayed [378, 596, 455, 191, 525, 616, 469, 647, 263, 442]
Displaying 10 samples:

```

![image-20250408162154069](imgs/taskB.png)
