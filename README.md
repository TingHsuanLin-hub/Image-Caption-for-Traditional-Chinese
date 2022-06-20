# Image-Caption-for-Traditioonal-Chinese

## Usage 
First, clone the repository locally:
```
$ git clone https://github.com/saahiluppal/catr.git
```
Then, install PyTorch 1.5+ and torchvision 0.6+ along with remaining dependencies:
```
$ pip install -r requirements.txt
```
That's it, should be good to train and test caption models.

## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  val2017/      # train and val images both place in this folder
```

## Training
Tweak the hyperparameters from <a href='https://github.com/saahiluppal/catr/blob/master/configuration.py'>configuration</a> file.

To train on a single GPU for 100 epochs run:
```
$ python main.py
```
We train with AdamW setting learning rate in the transformer to 1e-4.
Horizontal flips, scales an crops are used for augmentation.
Images are rescaled to have max size 256.
The transformer is trained with dropout of 0.1, and the whole model is trained with grad clip of 0.1.

## Testing
To test with your own images.
```
$ python predict.py --path /path/to/image --checkpoint /path/to/checkpoint
```

## pretrained model download:
<a href ='https://drive.google.com/drive/folders/1gAq4f_kPqdA_ygouYjYY7dJA8aCGvrCp?usp=sharing'> Image Caption Pre-trained Model</a>
Swin Transformer V2:   # Place in models/

Word2Vec:   # Place in Word2Vec_model/
            # 256, 512 is embedding size

