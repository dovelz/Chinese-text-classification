# Chinese Text Classification with CNN and RNN
<b> Update (January 14, 2020)</b> Pytorch implementation of [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882).

<br/>

<br/>


## References
pytorch implementation: https://github.com/songyingxin/TextClassification-Pytorch

Another pytorch implementation: https://github.com/galsang/CNN-sentence-classification-pytorch

<br/>


## Getting Started

### Requirements

python 3.7
pytorch 1.1
tqdm
sklearn

### Getting Data and Pretrained Word2vec
Data: http://thuctc.thunlp.org/message（THUCNews）

Pretrainded Word2vec: https://pan.baidu.com/s/14k-9jsspp43ZhMxqPmsWMQ（sogou pretrained）

More Chinese Word Vectors: https://github.com/Embedding/Chinese-Word-Vectors

 - Put the pretrained file in /dataTest/data/


###

<br>

### Train the model

To train the model, run command below.If you want change another model, edit the model in models directory and change the default in train.py.

```bash
$ python train.py
```
<br>

<br>

### Evaluate the model

Edit test.txt and run eval.py.

```bash
$ python eval.py
```

<br/>

## Results
<br/>

#### CNN3

```
Epoch [5/30]
Iter:    500,  Train Loss:  0.62,  Train Acc: 79.69%,  Val Loss:   0.8,  Val Acc: 77.40%,  Time: 0:00:58 *
Epoch [10/30]

Iter:   1100,  Train Loss:  0.28,  Train Acc: 92.19%,  Val Loss:  0.62,  Val Acc: 81.20%,  Time: 0:02:07 *
Epoch [15/30]

Iter:   1700,  Train Loss:  0.23,  Train Acc: 95.31%,  Val Loss:  0.57,  Val Acc: 82.00%,  Time: 0:03:14 *
Epoch [20/30]

Iter:   2300,  Train Loss:  0.13,  Train Acc: 100.00%,  Val Loss:  0.54,  Val Acc: 83.30%,  Time: 0:04:21 *
Epoch [25/30]

Iter:   2900,  Train Loss:  0.06,  Train Acc: 100.00%,  Val Loss:  0.55,  Val Acc: 82.60%,  Time: 0:05:29
Epoch [30/30]

Iter:   3500,  Train Loss: 0.033,  Train Acc: 100.00%,  Val Loss:  0.55,  Val Acc: 82.60%,  Time: 0:06:37
Test Loss:  0.55,  Test Acc: 82.28%


               precision    recall  f1-score   support

      finance     0.8523    0.7732    0.8108        97
       realty     0.8926    0.8852    0.8889       122
       stocks     0.7419    0.7188    0.7302        96
    education     0.8953    0.8462    0.8701        91
      science     0.7228    0.7604    0.7411        96
      society     0.8241    0.8318    0.8279       107
     politics     0.7857    0.7938    0.7897        97
       sports     0.8305    0.9800    0.8991       100
         game     0.8400    0.8235    0.8317       102
    entertainment     0.8372    0.7912    0.8136        91

     accuracy                         0.8228       999
    macro avg     0.8222    0.8204    0.8203       999
 weighted avg     0.8238    0.8228    0.8223       999
```

#### RNN3
```
Epoch [5/25]
Iter:    500,  Train Loss:   1.7,  Train Acc: 76.56%,  Val Loss:   1.7,  Val Acc: 74.20%,  Time: 0:00:22 *

Epoch [10/25]
Iter:   1100,  Train Loss:   1.5,  Train Acc: 92.97%,  Val Loss:   1.7,  Val Acc: 78.50%,  Time: 0:00:48

Epoch [15/25]
Iter:   1700,  Train Loss:   1.6,  Train Acc: 90.62%,  Val Loss:   1.7,  Val Acc: 79.20%,  Time: 0:01:14 *

Epoch [20/25]
Iter:   2300,  Train Loss:   1.5,  Train Acc: 92.97%,  Val Loss:   1.7,  Val Acc: 79.90%,  Time: 0:01:40

Epoch [25/25]
Iter:   2900,  Train Loss:   1.5,  Train Acc: 93.75%,  Val Loss:   1.7,  Val Acc: 79.30%,  Time: 0:02:06

Test Loss:   1.7,  Test Acc: 78.18%
```

#### RNN2
```
Epoch [10/50]
Iter:   1200,  Train Loss:  0.53,  Train Acc: 85.16%,  Val Loss:  0.74,  Val Acc: 76.80%,  Time: 0:01:24 *

Epoch [20/50]
Iter:   2300,  Train Loss:  0.39,  Train Acc: 86.72%,  Val Loss:  0.67,  Val Acc: 77.90%,  Time: 0:02:39

Epoch [30/50]
Iter:   3500,  Train Loss:  0.17,  Train Acc: 95.31%,  Val Loss:  0.64,  Val Acc: 80.10%,  Time: 0:04:03

Epoch [40/50]
Iter:   4700,  Train Loss:  0.12,  Train Acc: 98.44%,  Val Loss:  0.69,  Val Acc: 79.70%,  Time: 0:05:26

Epoch [50/50]
Iter:   5800,  Train Loss: 0.036,  Train Acc: 100.00%,  Val Loss:  0.69,  Val Acc: 79.50%,  Time: 0:06:43

Test Loss:  0.66,  Test Acc: 79.78%
```

#### RNN1
```
Epoch [5/15]
Iter:    500,  Train Loss:  0.51,  Train Acc: 83.59%,  Val Loss:  0.79,  Val Acc: 76.00%,  Time: 0:00:35 *

Epoch [10/15]
Iter:   1100,  Train Loss:  0.12,  Train Acc: 96.88%,  Val Loss:  0.78,  Val Acc: 79.60%,  Time: 0:01:16

Epoch [15/15]
Iter:   1700,  Train Loss: 0.0083,  Train Acc: 100.00%,  Val Loss:  0.86,  Val Acc: 81.30%,  Time: 0:01:57

Test Loss:  0.74,  Test Acc: 79.98%
```
