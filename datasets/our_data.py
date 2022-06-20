from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

from PIL import Image
import numpy as np
import random
import os


from .utils import  nested_tensor_from_tensor_list,read_json, Word2vec_addon

MAX_DIM = 256


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    # image = image.resize(new_shape)
    image = image.resize((MAX_DIM, MAX_DIM))
    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


train_transform = tv.transforms.Compose([
    RandomRotation(),
    tv.transforms.Lambda(under_max),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class CocoCaption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training', word2vec_model='../Word2Vec_model/word2vec_512.model'):
        super().__init__()

        self.root = root
        self.transform = transform
        self.annot = [(self._process_image(val['image_id']), (val['caption']))
                      for val in ann]
        if mode == 'validation':
            self.annot = self.annot
        if mode == 'training':
            # self.annot = self.annot[: limit]
            self.annot = self.annot

        # self.tokenizer = word2vec.Word2Vec.load(word2vec_model)
        self.tokenizer = Word2vec_addon(word2vec_model)
        self.max_length = max_length + 1

    def _process_image(self, image_id):
        # 讀取檔案名稱
        val = str(image_id)
        return val

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id))

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))
        

        # caption_encoded = self.tokenizer.encode_plus(
        #     caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)
        caption_encoded = self.tokenizer.encode_plus(
            caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True)

        # BERT model 的 encode_plus 所產生的 caption_encoded預設有 'input_ids', 'attention_mask'
        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
            1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return  image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


def build_dataset(config, mode='training'):
    # train dataset
    if mode == 'training':
        train_dir = os.path.join(config.dir, 'val_images')
        train_file = os.path.join(
            config.dir, 'annotations', 'captions_val.json')
        data = CocoCaption(train_dir, read_json(
            train_file), max_length=config.max_position_embeddings, limit=config.limit, transform=train_transform, mode='training', word2vec_model=config.wor2vec_model)
        return data

    # validation dataset
    elif mode == 'validation':
        val_dir = os.path.join(config.dir, 'val_images')
        val_file = os.path.join(
            config.dir, 'annotations', 'captions_train.json')
        data = CocoCaption(val_dir, read_json(
            val_file), max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform, mode='validation', word2vec_model=config.wor2vec_model)
        return data
    
    elif mode == 'test':
        val_dir = os.path.join(config.dir, 'val_images')
        val_file = os.path.join(
            config.dir, 'annotations', 'captions_test.json')
        data = CocoCaption(val_dir, read_json(
            val_file), max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform, mode='validation', word2vec_model=config.wor2vec_model)
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")
