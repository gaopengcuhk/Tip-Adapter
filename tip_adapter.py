import PIL
from PIL import Image
from typing import List, Dict

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset as TorchDataset

from . import clip
from .utils import *


class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, input_size, transform=None, is_train=False,
                 return_img0=False, k_tfm=1):
        self.data_source = data_source
        self.transform = transform # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = k_tfm if is_train else 1
        self.return_img0 = return_img0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                'Cannot augment the image {} times '
                'because transform is None'.format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = transforms.InterpolationMode.BICUBIC
        to_tensor = []
        to_tensor += [transforms.Resize(input_size, interpolation=interp_mode)]
        to_tensor += [transforms.ToTensor()]
        normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        )
        to_tensor += [normalize]
        self.to_tensor = transforms.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            'label': item["label"],
            'domain': item["domain"],
            'img': item["image"]
        }

        img0 = item["image"]

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = 'img'
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output['img'] = img

        if self.return_img0:
            output['img0'] = self.to_tensor(img0)

        return output['img'], output['label']

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img
    

class TipAdapter:
    def __init__(
        self, 
        device: str,
        template: List[str] = ['a photo of a {}.'],
        clip_backbone: str = "RN50",
        augment_epoch: int = 10,
        alpha: float = 1.,
        beta: float = 5.,
    ) -> None:
        """ Class that runs tip adapter
        Args: 
            device: specifying on which device to store tensors
            template: list of strings specifying what to prepend clip token embedding with
            clip_backbone: string indicating backbone to use for clip model (default is ResNet-50)
            augment_epoch: number of epochs to apply augmentation?? just use default...
        """

        self._device = device
        self._template = template
        self._augment_epoch = augment_epoch
        self._class_names = list()
        self._cache_keys = None
        self._cache_values = None
        self._clip_weights = None

        self._alpha = alpha
        self._beta = beta

        self._clip_model, self._preprocess = clip.load(clip_backbone, device=device)
        self._clip_model.eval()

        self._train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])


    def create_cache(
        self, 
        few_shot_data: Dict[str, List[PIL.Image.Image]],
    ) -> None:
        """ Creates the KV-cache
        Args: 
            few_shot_data: a dictionary where each key is the label and the value is a list of
                `PIL.Image` corresponding to an example image. The `PIL.Image` objects are assumed
                to be `RGB`, not `RGBA` (note make sure to do `.convert('RGB')`)      
        """
        # store class names
        self._class_names = list(few_shot_data.keys())
        # get clip weights
        self._clip_weights = clip_classifier(
            self._class_names, 
            self._template, 
            self._clip_model,
            self._device
        )

        cache_keys = list()
        cache_values = list()

        data_source = self._convert_data_dict_to_list(few_shot_data)
        train_loader_cache = self._build_data_loader(
            data_source=data_source,
            batch_size=256,
            tfm=self._train_tranform, 
            is_train=True, 
            shuffle=False
        )

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(self._augment_epoch):
                train_features = list()

                print('Augment Epoch: {:} / {:}'.format(augment_idx, self._augment_epoch))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.to(self._device)
                    image_features = self._clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.to(self._device)
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0))

        if "cuda" in self._device:
            cache_values = cache_values.half()
        elif "cpu" in self._device:
            cache_values = cache_values.float()

        self._cache_keys = cache_keys
        self._cache_values = cache_values

    def run(self, imgs: List[PIL.Image.Image]) -> List[str]:
        features = self._get_test_features(imgs)

        clip_logits = 100. * features @ self._clip_weights

        affinity = features @ self._cache_keys
        cache_logits = ((-1) * (self._beta - self._beta* affinity)).exp() @ self._cache_values

        tip_logits = clip_logits + cache_logits * self._alpha

        predicted_labels = torch.argmax(tip_logits, dim=1).cpu().numpy()

        predicted_classes = list()
        for label in predicted_labels:
            predicted_classes.append(self._class_names[label])

        return predicted_classes
       
    @property
    def cache(self) -> Dict[str, torch.Tensor]:
        '''Returns the cache values
        '''
        return {
            "keys": self._cache_keys, 
            "values": self._cache_values
        }

    @cache.setter
    def cache(self, cache: Dict[str, torch.Tensor]):
        '''Sets the cache values
        '''
        try:
            self._cache_keys = cache["keys"].to(self._device)
            self._cache_values = cache["values"].to(self._device)
        except KeyError as e:
            raise ValueError(f"Missing key in cache dictionary: {e}")


        if "cuda" in self._device:
            self._cache_values = self._cache_values.half()
        elif "cpu" in self._device:
            self._cache_values = self._cache_values.float()

    def load_cache_values(
        self, 
        file_path: str
    ):
        """Loads the cache from a file
        """
        try:
            loaded_data = torch.load(file_path)
            self.cache = loaded_data
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {file_path} was not found.")
        
    def save_cache_values(self, file_path: str):
        """Saves the cache values to a file."""
        cache_data = {
            "keys": self._cache_keys,
            "values": self._cache_values
        }
        torch.save(cache_data, file_path)


    def _get_test_features(self, imgs: List[PIL.Image.Image]) -> torch.Tensor:
        # for now just create dict with dummy labels
        few_shot_data = {self._class_names[0]: imgs}
        data_source = self._convert_data_dict_to_list(few_shot_data)
        test_loader_cache = self._build_data_loader(
            data_source=data_source,
            batch_size=256,
            tfm=self._preprocess, 
            is_train=False, 
            shuffle=False
        )

        features = list()

        with torch.no_grad():
            for images, _ in tqdm(test_loader_cache):
                images = images.to(self._device) 
                image_features = self._clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
        
        features = torch.cat(features)

        return features

    def _convert_data_dict_to_list(self, data_source: Dict[str, List[PIL.Image.Image]]):
        data_list = list()
        for idx, class_name in enumerate(self._class_names):
            if class_name not in data_source.keys():
                continue
            pil_image_list = data_source[class_name]
            for img in pil_image_list:
                data_list.append({
                    "image": img,
                    "label": idx,
                    "domain": -1,
                    "classname": class_name
                })

        return data_list
    
    def _build_data_loader(
        self,
        data_source=None,
        batch_size=64,
        input_size=224,
        tfm=None,
        is_train=True,
        shuffle=False,
        dataset_wrapper=None
    ):

        if dataset_wrapper is None:
            dataset_wrapper = DatasetWrapper

        # Build data loader
        data_loader = torch.utils.data.DataLoader(
            dataset_wrapper(data_source, input_size=input_size, transform=tfm, is_train=is_train),
            batch_size=batch_size,
            num_workers=8,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=(torch.cuda.is_available())
        )
        assert len(data_loader) > 0

        return data_loader
    

# Sanity test
if __name__ == "__main__":
    import glob
    import os

    tip_adapter = TipAdapter(device="cuda")
    # replace this with data dir
    test_data_dir = "/home/ritviks/git/Tip-Adapter/data/caltech-101/101_ObjectCategories"
    folders = glob.glob(test_data_dir + "/*")[:10]
    data = dict()

    test_data = dict()
    for folder in folders:
        imgs = glob.glob(folder+"/*")
        label = os.path.basename(folder)

        num_train = int(len(imgs) * 0.8)
        num_test = len(imgs) - num_train
        
        train_ims = list()
        test_ims = list()

        for img in imgs[:num_train]:
            train_ims.append(Image.open(img).convert("RGB"))

        for img in imgs[num_train:]:
            test_ims.append(Image.open(img).convert("RGB"))

        data[label] = train_ims
        test_data[label] = test_ims
    
    tip_adapter.create_cache(data)
    tip_adapter.save_cache_values("cache_values.pt")

    tip_adapter.load_cache_values("cache_values.pt")

    cache_values = tip_adapter.cache
    tip_adapter.cache = cache_values

    for label, im_list in test_data.items():
        print("Label: ", label)
        predicted_classes = tip_adapter.run(im_list)
        print("Predicted classes: ", predicted_classes)
