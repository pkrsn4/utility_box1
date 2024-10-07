import os
import numpy as np

from tqdm.auto import tqdm

import albumentations as A

from torch.utils.data import Dataset as BaseDataset
import torch
    
def get_training_augmentation(input_image_shape):
    scale = 0.25 #Maximum Should be 0.5, Downscale
    
    scale_setting = 0.25 #Color Jitter
    scale_color = 0.1
    
    aug_list = [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Blur(p=0.1, blur_limit=9),
        A.GaussNoise(p=0.2, var_limit=10),
        A.ColorJitter(p=0.2,brightness=scale_setting,contrast=scale_setting,saturation=scale_color,hue=scale_color / 2,),
        A.Superpixels(p=0.2,p_replace=0.1,n_segments=10000,max_size=int(input_image_shape / 2),),
        A.ZoomBlur(p=0.2, max_factor=1.05),
        A.RandomBrightnessContrast(p=0.4),
    ]
    train_aug = A.Compose(aug_list)
    return train_aug

class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
        

class CustomDataset(BaseDataset):
    
    def __init__(self, image_dir, mask_dir, trial_run = False ,dist_df = None, total_samples = None ,augmentation=None):
        
        assert len(os.listdir(image_dir)) == len(os.listdir(mask_dir))
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.aug = augmentation
        
        self.dist_df = dist_df
        
        if self.dist_df is not None and total_samples is None:
            raise CustomError("If Distribution DataFrame is Provided, Total Samples can't be None")
        
        if self.dist_df is not None:
            
            self._gc_samples = list(self.dist_df[dist_df['gc']>0]['patch_name'])
            self._non_gc_samples = list(self.dist_df[dist_df['gc'] == 0]['patch_name'])
            self._gc_probability = len(self._gc_samples)/(len(self._gc_samples)+len(self._non_gc_samples))
            self.sample_strategy = True #sample_strategy has no use, may be in future
            self.gc_counter = 0
            self.non_gc_counter = 0
            
            images = []
            for idx in tqdm(range(total_samples), desc='Sampling'):
                try:
                    images.append(self._generate_sample())
                except Exception as e:
                    #print(e)
                    continue
                    
            
            images_npy = [f"{image}.npy" for image in images]
            self.images = images_npy
            self.total_samples = len(self.images)
            print(f"Total Images Sampled: {self.total_samples}")
        elif trial_run:
            self.images = os.listdir(image_dir)[:100]
            self.total_samples = len(self.images)
        
        else:
            self.images = os.listdir(image_dir)
            self.sample_strategy = False
            self.total_samples = len(self.images)

        
        #mean = (0.485, 0.456, 0.406)
        #std = (0.229, 0.224, 0.225)
        
        #mean = (0.5, 0.5, 0.5)
        #std = (0.5, 0.5, 0.5)
        #self.normalize= transforms.Normalize(mean=mean, std=std)
    
    def __getitem__(self, index):
        

        image_name = self.images[index]
        image = np.load(f'{self.image_dir}/{image_name}').astype(np.uint8)
        mask =  np.load(f'{self.mask_dir}/{image_name}')
        
        mask[mask == 2] = 0
        
        if self.aug:
            aug_data = self.aug(image=image, mask=mask)
            image, mask = aug_data['image'], aug_data['mask']
            
        image = image.transpose((2, 0, 1))
        image = torch.as_tensor(image).float().contiguous()
        image = image/255
        return image, mask
    
    def __len__(self):
            return self.total_samples
    
    def _generate_sample(self):
        if random.uniform(0,1)>self._gc_probability:
            if random.uniform(0,1)> self._gc_probability:
                image_name = random.sample(self._non_gc_samples, 1)[0]
                self._non_gc_samples.remove(image_name)
                self.non_gc_counter += 1
            else:
                image_name = random.sample(self._gc_samples, 1)[0]
                self._gc_samples.remove(image_name)
                self.gc_counter += 1
        else:
            image_name = random.sample(self._gc_samples, 1)[0]
            self._gc_samples.remove(image_name)
            self.gc_counter += 1
    
        return image_name