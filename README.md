# Readme
The scripts are for fine-tuning using pre-trained Alexnet in matconvnet.

Currently the official tutorial and the example codes do not provide any information on how to finetune the pre-trained cnn. After working two days, I finally figured it out.

Since the preparating of data in matconvnet is a little complicated, unlike Caffe, in "setup_data.m" file I particularly simplies the data preparation.

## data preparation
prepare data similar to Caffe under <dataPath> directory

	<dataPath>/train.txt

	<dataPath>/test.txt

	<dataPath>/category.txt

In train.txt and test.txt, each line is a pair: <imagePath> <label>, label starts from 1
e.g.: 

 /home/experiment/data/cat001.jpg 2

 /home/experiment/data/dog011.jpg 1

In category.txt, each line is the label text, and first line correspond to label 1, second line label 2, and so on. For example, in category.txt:

dog

cat

rice

then, dog is label 1, cat is label 2, rice is label 3, etc.

## finetune
The finetune function are as follows:

[net, info] = finetune_alexnet(dataPath, varargin)

dataPath has to be specified in the way as above. additional argument can be passed but not necessary.

Put the scripts at the root folder of <matconvnet>

put pretrained alexnet: imagenet-matconvnet-alex.mat under the same directory.

The script is adapted from matconvnet/example/imagenet/cnn_imagenet.m

Just prepare the data properly and call this function, finetuning will starts.

Basically, what I did is just call cnn_imagenet_init to get alexnet and overwrite the weights from pretrained alexnet and adapt the final two layers (fc8 and softmax) to customized dataset.

As I tested, finetuning from pretrained alexnet converges faster than from scratch.

Feel free to change to finetune over nets.