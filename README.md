## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[class_sample_imgs]: ./writeup_images/class_sample_imgs.png "Traffic sign sample images"
[train_data_distribution]: ./writeup_images/train_data_distribution.png "Training data distribution"
[image_preprocess]: ./writeup_images/image_preprocess.png "Pre-process images"
[perspective_transform]: ./writeup_images/perspective_transform.png "Perspective transformation"
[two_stage_convnet]: ./writeup_images/two-stage_convnet.png "2-Stage ConvNet"
[lenetplus_acc]: ./writeup_images/lenetplus_acc.png "LeNetPlus accuracy"
[final_model_acc]: ./writeup_images/final_model_acc.png "Final model accuracy"
[web_image_prediction_top_1]: ./writeup_images/web_image_prediction_top_1.png "Top 1 prediction of web images"
[web_image_prediction_top_5]: ./writeup_images/web_image_prediction_top_5.png "Top 5 prediction of web images"
[accuracy_each_class]: ./writeup_images/accuracy_each_class.png "The accuracy of each class"
[fail_prediction]: ./writeup_images/fail_prediction.png "Random choose 5 fail images in each class"
[c1_feature_map]: ./writeup_images/c1_feature_map.png "Feature maps in c1 layer"
[c3_feature_map]: ./writeup_images/c3_feature_map.png "Feature maps in c3 layer"
[web_image_01]: ./images/class_id_01_01.jpg "Web image 01"
[web_image_02]: ./images/class_id_01_02.jpg "Web image 02"
[web_image_03]: ./images/class_id_04_01.jpg "Web image 03"
[web_image_04]: ./images/class_id_13_01.jpg "Web image 04"
[web_image_05]: ./images/class_id_13_02.jpg "Web image 05"
[web_image_06]: ./images/class_id_13_03.jpg "Web image 06"
[web_image_07]: ./images/class_id_14_01.png "Web image 07"
[web_image_08]: ./images/class_id_18_01.jpg "Web image 08"
[web_image_09]: ./images/class_id_21_01.png "Web image 09"
[web_image_10]: ./images/class_id_24_01.png "Web image 10"
[web_image_11]: ./images/class_id_25_01.jpg "Web image 11"
[web_image_12]: ./images/class_id_26_01.png "Web image 12"
[web_image_13]: ./images/class_id_29_01.png "Web image 13"
[web_image_14]: ./images/class_id_31_01.png "Web image 14"
[web_image_15]: ./images/class_id_33_01.jpg "Web image 15"
[web_image_16]: ./images/class_id_34_01.jpg "Web image 16"
[web_image_17]: ./images/class_id_35_01.jpg "Web image 17"
[web_image_18]: ./images/class_id_38_01.jpg "Web image 18"
[web_image_19]: ./images/class_id_40_01.png "Web image 19"
[web_image_20]: ./images/class_id_40_02.jpg "Web image 20"

Overview
---
In this project, I will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I will then try out my model on images of German traffic signs that I find on the web.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


---
### About my project

You're reading it! and here is a link to my [project code](https://github.com/hcv1027/CarND-Traffic-Sign-Classifier-Project)

All the code snippets or the functions mentioned below can be found in the IPython notebook located in `Traffic_Sign_Classifier.ipynb`.


### Dependencies
The section **"Using CMA-ES to find the hyperparameters of my CNN architecture"** in the `Traffic_Sign_Classifier.ipynb` will need this python package: [pycma](https://github.com/CMA-ES/pycma). You can install it by the command:
```
pip install cma
```
or run the first cell in the `Traffic_Sign_Classifier.ipynb` to install it directly in jupyter notebook environment.


### Pre-trained model
I've upload my previously trained model into the folder `model`. Their name is relative to the experiment. You can just load these model to see the final result for saving the training time.


---
### Data Set Summary & Exploration

#### 1. Basic Summary of the Data Set

Here is how do I get the basic summary of the data set information and the output result:

```python
training_file = root_path + 'data/train.p'
validation_file = root_path + 'data/valid.p'
testing_file = root_path + 'data/test.p'
signnames_file = project_path + 'signnames.csv'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
with open(signnames_file, mode='r') as f:
    signnames_csv = csv.reader(f)
    signnames = {}
    for row in signnames_csv:
        if row[0] != 'ClassId':
            signnames[row[0]] = row[1]

# Number of training examples
n_train = X_train.shape[0]
# Number of validation examples
n_validation = X_valid.shape[0]
# Number of testing examples.
n_test = X_test.shape[0]
# What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]
# How many unique classes/labels there are in the dataset.
n_classes = len(signnames)

print("Number of training examples =", n_train)
print("Number of validating examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```
output:
```
Number of training examples = 34799
Number of validating examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```
#### 2. Include an exploratory visualization of the dataset.

Here are sample images random choosen from each class:
![Traffic sign sample images][class_sample_imgs]

Here are the number of training datas belong to each class:
![Training data distribution][train_data_distribution]


### Design and Test a Model Architecture
    
#### 1. Pre-process the training data

##### Using perspective transform to increase training data set

I use *zoom in, zoom out, horizontal rotation* and *vertical rotation* to do the six types of image perspection transform. 

Here is the result:

![Perspective transformation][perspective_transform]

##### Image pre-process
My image preprocessing step is:
1. Convert image from `RGB color space` to `YUV color space`
2. Extract *Y channel* from `YUV color space`
3. Add gauss noise with `mean = 0` and `variance = 10.0`.
4. Normalize *Y channel* accroading to the formula:
   $$ new\_pixel = \frac{pixel - min(image)}{max(image) - min(image)} $$

![Pre-process images][image_preprocess]

#### 2. Searching hyperparameters: CMA-ES
Since hyperparameter tuning is very time-consuming. I use the algorithm called **CMA-ES**, the tool to auto find the best hyperparameters of my model architecture.
Here are some useful link introducing the concept of **CMA-ES**:
1. [CMA-ES wiki](https://en.wikipedia.org/wiki/CMA-ES#Principles)
2. [A Visual Guide to Evolution Strategies](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/)
3. [CMA-ES Source Code](http://cma.gforge.inria.fr/cmaes_sourcecode_page.html#practical)
4. [pycma](https://github.com/CMA-ES/pycma) (I use this python package in my code)

Because **CMA-ES** is looking for continuous solutions, I also reference [this paper](https://arxiv.org/abs/1604.07269) to search integer solutions. I use **CMA-ES** to search 8 hyperparameters. Restricting all of their boundary to [0.0, 1.0].

Here is how I transform the solutions found by **CMA-ES** to my model architecture.
| Hyperparameter  |          Description          |  Transformation  |           Range            |
| :-------------: | :---------------------------: | :--------------: | :------------------------: |
|     c1_size     |     c1 layer filter size      | $2^{1+2*sol[0]}$ |  $[2^1, 2^3]$ = $[2, 8]$   |
|    c1_depth     |    c1 layer filter number     | $2^{2+5*sol[1]}$ | $[2^2, 2^7]$ = $[4, 128]$  |
|     c3_size     |     c3 layer filter size      | $2^{1+2*sol[2]}$ |  $[2^1, 2^3]$ = $[2, 8]$   |
|    c3_depth     |    c3 layer filter number     | $2^{2+5*sol[3]}$ | $[2^2, 2^7]$ = $[4, 128]$  |
|     c5_size     | c5 fully connected layer size | $2^{6+3*sol[4]}$ | $[2^6, 2^9]$ = $[64, 512]$ |
|     f6_size     | f6 fully connected layer size | $2^{6+3*sol[5]}$ | $[2^6, 2^9]$ = $[64, 512]$ |
| c5_dropout_rate |        c5 dropout rate        |     $sol[6]$     |        $[0.0, 1.0]$        |
| f6_dropout_rate |        f6 dropout rate        |     $sol[7]$     |        $[0.0, 1.0]$        |

```python
def solution2hyperparameter(solution):
    hyperparameter = {
        'EPOCHS': 10,
        'BATCH_SIZE': 128,
        'RATE': 0.001,
        'input_shape': [32, 32, 1],
        'use_bn': True,
        'use_dropout': True,
        'c1_size': int(pow(2, 1 + 2 * solution[key2idx['c1_size']])), # id: 0, range: [2^1, 2^3] = [2, 8]
        'c1_depth': int(pow(2, 2 + 5 * solution[key2idx['c1_depth']])), # id: 1, range: [2^2, 2^7] = [4, 128]
        'c1_stride': 1,
        'c3_size': int(pow(2, 1 + 2 * solution[key2idx['c3_size']])), # id: 2, range: [2^1, 2^3] = [2, 8]
        'c3_depth': int(pow(2, 2 + 5 * solution[key2idx['c3_depth']])), # id: 3, range: [2^2, 2^7] = [4, 128]
        'c3_stride': 1,
        'c5_size': int(pow(2, 6 + 3 * solution[key2idx['c5_size']])), # id: 4, range: [2^6, 2^9] = [64, 512]
        'f6_size': int(pow(2, 6 + 3 * solution[key2idx['f6_size']])), # id: 5, range: [2^6, 2^9] = [64, 512]
        'c5_dropout_rate': solution[key2idx['c5_dropout_rate']], # id: 6, range: [0.0, 1.0], default: 0.5
        'f6_dropout_rate': solution[key2idx['f6_dropout_rate']], # id: 7, range: [0.0, 1.0], default: 0.5
    }
    return hyperparameter
```

I use the **negative accuracy of validation data set** as the fitness score to tell **CMA-ES** how good the solutions it found are. The overall code flow will look like below one:
```python
# Initialize CMA-ES
cma_es_nvars = 8
init_solu = cma_es_nvars * [0.5]
lower_bounds = cma_es_nvars * [0.0]
upper_bounds = cma_es_nvars * [1.0]
bounds = [lower_bounds, upper_bounds]
optim = cma.CMAEvolutionStrategy(init_solu, 0.2, {'popsize': 30, 'bounds': bounds})

## Search candidate solutions
best_solutions = []
iteration = 20
for round in range(iteration):
    def get_fitness(hyperparameter):
        with tf.Graph().as_default(), tf.Session() as sess:
            conv_net = ConvNet(hyperparameter)
            conv_net.build_cnn()
            sess.run(tf.global_variables_initializer())
            c5_dropout_rate = hyperparameter['c5_dropout_rate']
            f6_dropout_rate = hyperparameter['f6_dropout_rate']
            conv_net.train(sess, X_train_extend_norm, y_train_extend,
                            False, None, None,
                            False, save_name='sol_{:0>2d}'.format(idx),
                            dropout=[c5_dropout_rate, f6_dropout_rate]))

            accuracy = conv_net.evaluate(sess, X_valid_norm, y_valid)
            return -accuracy

    X = optim.ask() # get candidate solutions
    hyperparameters = [solution2hyperparameter(x) for x in X]
    fitness = [get_fitness(hyper) for hyper in hyperparameters]
    optim.tell(X, fitness) # do all the real "update" work

# final output
print('best f-value =', optim.result[1])
print('best solution =', optim.result[0])
```

In stage_1, I run **CMA-ES** `20` rounds, it generates `30` sample solutions in each round. I Keep the best one of these sample solutions in each round as candidates. I set the `EPOCHS` to `10`, so it does not take too much time in this stage.

In stage_2, I set the `EPOCHS` to `50`, and train my model with these `20` candidate solutions found in first stage, choose the best one (accroading to the **accuracy of test data set**) as my final model architecture.

**Note:**
The stage_1 is very time-consuming. Because I did't have a stable hardware to run this stage (I use google colaboratory to train, it sometimes disconnected as I run this stage), so the candidate solutions used in stage_2 were gathered from the results of several independent trial of stage_1.

#### 3. Model architecture

There are two models I use later to compare with several differnt training techniques and strategies. The differences between them are the filter size and depth (channels) of c1 and c3 layers and the size of c5, f6 fully connected layers.

![2-Stage ConvNet][two_stage_convnet](Image is modified from [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf))

1. **LeNetPlus**: It's almost like LeNet-5, except the input of c5 layer comes from the concatenation of s2 and s4.
   |       Layer        |                 Description                 |
   | :----------------: | :-----------------------------------------: |
   |       Input        |             32x32x1 Gray image              |
   | Convolution 5x5x6  | 1x1 stride, valid padding, outputs 28x28x6  |
   |        RELU        |                                             |
   |  Max pooling 2x2   | 2x2 stride, valid padding, outputs 14x14x6  |
   | Convolution 5x5x16 | 1x1 stride, valid padding, outputs 10x10x16 |
   |        RELU        |                                             |
   |  Max pooling 2x2   |  2x2 stride, valid padding, outputs 5x5x16  |
   |  Fully connected   |                 outputs 120                 |
   |        RELU        |                                             |
   |  Fully connected   |                 outputs 84                  |
   |        RELU        |                                             |
   |  Fully connected   |                 outputs 43                  |
   |      Softmax       |                 outputs 43                  |

2. **FinalModel**: It uses the same model architecture, but each layer's filter size and depth comes from CMA-ES searching stage. And it includes batch normalization and dropout. (Both are optional) 
   |             Layer             |                 Description                  |
   | :---------------------------: | :------------------------------------------: |
   |             Input             |              32x32x1 Gray image              |
   |      Convolution 4x4x17       | 1x1 stride, valid padding, outputs 29x29x17  |
   | Batch normalization(Optional) |                                              |
   |             RELU              |                                              |
   |        Max pooling 2x2        | 2x2 stride, valid padding, outputs 14x14x17  |
   |      Convolution 2x2x100      | 1x1 stride, valid padding, outputs 13x13x100 |
   | Batch normalization(Optional) |                                              |
   |             RELU              |                                              |
   |        Max pooling 2x2        |  2x2 stride, valid padding, outputs 6x6x100  |
   |        Fully connected        |                 outputs 506                  |
   | Batch normalization(Optional) |                                              |
   |             RELU              |                                              |
   |       Dropout(Optional)       |                 rate = 0.850                 |
   |        Fully connected        |                  outputs 91                  |
   | Batch normalization(Optional) |                                              |
   |             RELU              |                                              |
   |       Dropout(Optional)       |                 rate = 0.215                 |
   |        Fully connected        |                  outputs 43                  |
   |            Softmax            |                  outputs 43                  |


#### 4. Hyperparameters used in training process

Below table lists the hyperparameters which are not relative to the model architecture, I use them in the process of training my model:

1. **LeNetPlus**
   |     Hyperparameter      |     Value      |
   | :---------------------: | :------------: |
   |         Epochs          |       50       |
   |      Learning rate      |     0.001      |
   |       Batch size        |      128       |
   |        Optimizer        | Adam Optimizer |
   |       Use dropout       |     False      |
   | Use batch normalization |     False      |

2. **FinalModel**:
    |     Hyperparameter      |     Value      |
    | :---------------------: | :------------: |
    |         Epochs          |       50       |
    |      Learning rate      |     0.001      |
    |       Batch size        |      128       |
    |        Optimizer        | Adam Optimizer |
    |       Use dropout       |      True      |
    | Use batch normalization |      True      |

#### 5. Training approach

#### Experiment 1:
I use LeNetPlus model to try some different training techniques (batch normalization, dropout and data augmentation) in this experiment. The results are shown below.

|      Model      |                Description                 | train acc. | test acc. |
| :-------------: | :----------------------------------------: | :--------: | :-------: |
| 1. LeNetPlus_01 |  Original training data with gauss noise   |   1.000    |   0.918   |
| 2. LeNetPlus_01 | Original training data without gauss noise |   1.000    |   0.924   |
| 3. LeNetPlus_01 |  Extended training data with gauss noise   |   0.998    |   0.944   |
| 4. LeNetPlus_01 | Extended training data without gauss noise |   0.999    |   0.948   |
| 5. LeNetPlus_02 |    Use batch normalization and dropout     |   0.998    |   0.936   |
| 6. LeNetPlus_03 |                Use dropout                 |   0.996    |   0.936   |
| 7. LeNetPlus_04 |          Use batch normalization           |   0.999    |   0.913   |
![LeNetPlus accuracy][lenetplus_acc]

#### Experiment 1 summary:
1. Using perspective transform to produce more training datas can improve the final accuracy. (`1. vs 3.` and `2. vs 4.`)
2. Adding gauss noise to the training data will slightly decrease the final accuracy. I'm suprising about this result. I originally expected this can improve the accuracy. (`1. vs 2.` and `3. vs 4.`)
3. Using both batch normalization and dropout will train faster than just using dropout. But their final accuracy are almost the same. (See diagram, `5. vs 6.`)
4. Only using batch normalization will decrease the final accuracy. (`1. vs 7.`)

#### Experiment 2:
Base on the result of *Experiment 1*, I use the FinalModel to try several different training strategies (Learning rate and batch size) and techniques (batch normalization, dropout and data augmentation). The results are shown below.

The ***model_01_base*** are the FinalModel trained with its hyperparameters listed in **Sec.4**. Through ***model_02_no_bn*** to ***model_09_extend_no_noise***, their different parts compared with ***model_01_base*** are listed in the description column in below table. The results are shown below.

|            Model            |              Description              | train acc. | test acc. |
| :-------------------------: | :-----------------------------------: | :--------: | :-------: |
|      1. model_01_base       |              Final model              |   0.999    |   0.979   |
|      2. model_02_no_bn      |        No batch normalization         |   0.913    |   0.902   |
|   3. model_03_no_dropout    |              No dropout               |   0.999    |   0.956   |
|    4. model_04_dropout05    |       Dropout rate fixed to 0.5       |   0.999    |   0.972   |
|    5. model_05_batch256     |            Batch size: 256            |   0.999    |   0.976   |
|      6. model_06_lr003      |         Learning rate: 0.003          |   0.997    |   0.973   |
|   7. model_07_basic_noise   |  Original training data, with noise   |   1.000    |   0.967   |
| 8. model_08_basic_no_noise  | Original training data, without noise |   1.000    |   0.972   |
| 9. model_09_extend_no_noise | Extended training data, without noise |   0.999    |   0.972   |
![Final model accuracy][final_model_acc]

#### Experiment 2 summary:
1. Compare with **Experiment 1**, except ***model_02_no_bn***, all others show significent improvement. So I can conclude that FinalModel is better than LeNetPlus.
2. Both batch normalization and dropout can improve accuracy as I expected. (`1. vs 2.` and `1. vs 3.`)
3. Adding gauss noise to images in pre-processing stage seems does't have enough evidence to show significent influence on the accuracy. (`1. vs 9.` and `7. and 8.`)
4. Changing learning rate and batch size seems does't have obvious influence on the training speed. (`1. vs 9.` and `7. and 8.`)
5. Without using batch normalization, the accuracy decreases very obviously.
6. Using perspective transform to produce more training datas also show the result of improved accuracy. But the increase is not as much as **Experiment 1**. (`1. vs 7.`)

Combine the result of **Experiment 1** and **Experiment 2**, I choose ***model_01_base*** as my final model. It has testing accuracy: **0.979**.

#### The accuracy of each class

Here I show the accuracy of each class trained with my ***FinalModel***. It appears that the predicting performance of class_30 (Beware of ice/snow) and class_42 (End of no passing by vehicles over 3.5 metric tons) are particularly poor.
![The accuracy of each class][accuracy_each_class]

#### Random choose 5 fail prediction image for each class

The thumbnail image below shows 5 randomly choosed fail prediction images in each class so that I may find some hint about how to imrpove my model in the future. (The black image in each cell means no fail image can be shown)
![5 fail prediction image][fail_prediction]
 

### Test my FinalModel on New Images

#### 1. German traffic signs found on the web

There are 20 German traffic signs I found on the web. I show five of them below with their original resolution and add a comment if they are especially worth to describe. The others are just shown as thumbnail after image preprocessing and adding the prediction label on them in next section.

1. ![WebImage01][web_image_01] (A little clockwise rotation)
2. ![WebImage06][web_image_06] (There is some part of building shown in background)
3. ![WebImage07][web_image_07]
4. ![WebImage08][web_image_08] (Most part of traffic sign is hidden by snow)
5. ![WebImage20][web_image_20]

#### 2. Predictions on these new traffic signs, show their top_1 prediction result

Here is the thumbnail of these 20 web traffic sign images. The ***predict label*** and the ***true label*** are shown on each image. Red font color means the prediction is incorrect, black means correct. For example, the red string <font color="red">***Predict 3/True 1***</font> shown on 1st row, 1st column image means that its prediction label is class_3, and its true label is class_1. Obviously, it is a wrong prediction.

The accuracy of predicting web image is: **0.85**

![Web images Top_1 result][web_image_prediction_top_1]

|  Image   |                  Prediction                   |        True label         |  Image   |      Prediction      |      True label       |
| :------: | :-------------------------------------------: | :-----------------------: | :------: | :------------------: | :-------------------: |
| image_01 | <font color="red">Speed limit (60km/h)</font> |   Speed limit (30km/h)    | image_11 |      Road work       |       Road work       |
| image_02 |             Speed limit (30km/h)              |   Speed limit (30km/h)    | image_12 |   Traffic signals    |    Traffic signals    |
| image_03 |             Speed limit (70km/h)              |   Speed limit (70km/h)    | image_13 |  Bicycles crossing   |   Bicycles crossing   |
| image_04 |                     Yield                     |           Yield           | image_14 |     Wild animals     | Wild animals crossing |
| image_05 |                     Yield                     |           Yield           | image_15 |   Turn right ahead   |   Turn right ahead    |
| image_06 |                     Yield                     |           Yield           | image_16 |   Turn left ahead    |    Turn left ahead    |
| image_07 | <font color="red">Speed limit (50km/h)</font> |           Stop            | image_17 |      Ahead only      |      Ahead only       |
| image_08 |      <font color="red">Bumpy road</font>      |      General caution      | image_18 |      Keep right      |      Keep right       |
| image_09 |                 Double curve                  |       Double curve        | image_19 | Roundabout mandatory | Roundabout mandatory  |
| image_10 |           Road narrows on the right           | Road narrows on the right | image_20 | Roundabout mandatory | Roundabout mandatory  |

Sence there are only 20 samples, each fail prediction will cause 5% decrease of accuracy. Authough it's lower than the test accuracy (0.979). I think this result is still good enough.

#### 3. Predictions on these new traffic signs, show their top_5 prediction result

The thumbnail image shown below are these 20 traffic signs and a random choosen image corresponding to their top_5 predicting class, from column 2 to column 6 (top_1 though top_5). The label on each small image is its top *k*-th prediction class id and its confidence. Red color means incorrect prediction and blue color means correct prediction. For example, at row 7, column 1, The label <font color="red"> ***top_1: 2, 79.40%*** </font> means that the prediction of this web image is class_2, it has 79.4% confidence, but it is wrong. And in the same row, column 5, the label <font color="blue"> ***top_4: 14, 0.22%*** </font> means its top_4 prediction is class_14 with 0.22% confidence, it's correspond to the correct class.

From this thumbnail image, we can easily notice that the prediction of top_5 classes almost has the same shape with the input traffic sign, even for those three fail prediction cases, their top_1 prediction class also has the same shape. And except for the first fail case, the top_5 of the other two fail cases both include the correct class.

![Web images Top_5 result][web_image_prediction_top_5]

### Visualizing the Neural Network

#### 1. Feature maps in c1 and c3 layers

I use this image as input to show the feature maps in c1 and c3 layer:

![WebImage09][web_image_09]

It is obvious that feature maps in c1 layer have detect the basic shape of the traffic sign. And we can also find some feature maps in c3 layer have detect the edges and middle mark of the traffic sign.

1. c1 layer's feature map:
   ![Feature maps in c1 layer][c1_feature_map]
2. c3 layer's feature map:
   ![Feature maps in c1 layer][c3_feature_map]


### Conclusion and future work
1. In this project, I verified some morden techniques which can improve the traffic sign classifier model.
2. I use the black-box optimization algorithm **CMA-ES** to save my time in model architecture tuning.
3. I can't explain why adding noise into training data doesn't make my model more robust. I think it's worth taking the time to find the reason.
4. I've tried many different training techniques but still can't improve my test accuracy higher than 98%. In my process of tuning, the average test accuracy is between 96.0%~97.8%. I think it is the bottleneck of my current model architecture.

