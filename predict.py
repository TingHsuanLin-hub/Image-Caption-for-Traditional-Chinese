import torch

# from transformers import BertTokenizer
from gensim.models import word2vec
import numpy as np
from PIL import Image
import argparse

from models import our_caption
from datasets import our_data, utils
from configuration import Config
import os

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, help='path to image', required=True)
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
args = parser.parse_args()
image_path = args.path
checkpoint_path = args.checkpoint

config = Config()

print("Checking for checkpoint.")
if checkpoint_path is None:
    raise NotImplementedError('No model to chose from!')
else:
    if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
    print("Found checkpoint! Loading!")
    model,_ = our_caption.build_model(config)
    print("Loading Checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

tokenizer = word2vec.Word2Vec.load(config.wor2vec_model)
start_token = tokenizer.wv["*"]
end_token = tokenizer.wv["~"]


image = Image.open(image_path)
image = our_data.val_transform(image)
image = image.unsqueeze(0)


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length, tokenizer.vector_size), dtype=torch.float32)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0, :] = torch.tensor(start_token)
    mask_template[:, 0] = False

    return caption_template, mask_template


caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)


@torch.no_grad()
def evaluate():
    model.eval()
    result = []
    for i in range(config.max_position_embeddings - 1):
        predictions = model( image, caption, cap_mask)
        predictions = predictions[:, i, :]

        similar = tokenizer.wv.most_similar(predictions.numpy())
        # print(similar[0][1])
        word = similar[0][0]
        if word == "~":
            return caption, result
        # word = similar[0][0]

        caption[:, i+1,:] = torch.tensor(tokenizer.wv[word])
        cap_mask[:, i+1] = False
        result.append(word)

    return caption, result

output, result = evaluate()

result = "".join(result)
print(result)