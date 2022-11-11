# Kaggle
Practising my ML skill with Kaggle

### MNIST Handwritten Digits Classification

One of the well-known dataset for beginners. At this time I'm pretty new to the domain with no hands-on skill. The neural network architectures I used to tackle with this dataset is the VGG-like architectures (neural networks with blocks of Conv3x3-ReLU-BatchNorm for the feature extractors and the dense layers for classification).

This is the first time I experienced building neural networks on Google Colab, and with this small dataset, experimenting with different regularization techniques such as Dropout, BatchNorm, different activation functions (Tanh, Sigmoid or Softmax), using learning rate scheduler, hyperparameter tuning was really fun.

The datasets: http://yann.lecun.com/exdb/mnist/
VGG paper: https://arxiv.org/abs/1409.1556


### Bali26 Plants Dataset Classification

After a long time far away from Kaggle, I found a competitions held by The International Society of Data Scientists (ISODS) for image classification. Even though the competition was over by the time I found it, I still want to try something to measure my skills after a big gap (for my university courses and self research on Deep Learning field).

At first I attempted to train a model from scratch but I realised that the computation I had was limited (low-cost GPU like Google Colab), so I tried to
use transfer learning on pretrained MobileNetV3 with ImageNet1K weights but I had trouble debugging the model (some problems occur since this is also the first time I used Pytorch). So, I switched to using pretrained EfficientNetB7 with ImageNet1K weights. The convolutional layers were freeze as a feature extractor and the classification layer will be trained on the dataset.

Since it was the first time I used Pytorch, the experiences at the beginning were brutal :(. I had problems when using GPU (I use most of my time debugging on torch datatype, device problem when there were conflicts between GPU and CPU with my data, first time writing some customizable Dataset class, learn how to build a model using `torch.nn.Module`). But overall, Pytorch is fun to use. It makes me to write my own functions for the training and the validation steps, create my own progress bar to track the model learning progress and saving the model weights and hyperparameters (I wasted hours retraining the model because of my stupidity of not saving the model, typo issues).

After this mini-project, I learned a lot of how to use Python class, `Tuple`, `Dict`, `OrderedDict`, read more papers, become more active in learning new things. I think this is a big win for me :).

The competition: www.kaggle.com/competitions/classification-of-plants-of-southeast-asia/
The dataset: https://realtechsupport.org/projects/bali26_overview_bahasa.html
The network architecture that I used: https://arxiv.org/pdf/1905.11946.pdf
Some techniques that I find interesting during the project:
1. Data augmentation: https://arxiv.org/pdf/2103.10158.pdf
2. Cosine Annealing Learning Rate Scheduler: https://arxiv.org/pdf/1608.03983.pdf
3. Transfer Learning: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
4. Label Smoothing: https://arxiv.org/pdf/2011.12562.pdf

After this, I think I should put some work on NLP-related problems, since some of the models in Computer Vision is inspired by NLP and vice versa. Stay tuned :).
