import albumentations as A
from albumentations.pytorch import ToTensor

__all__ = ['factory', ]


def heavy_1(image_size, always_apply=False, p=0.5):
    tile_h, tile_w = image_size, image_size
    crop_min, crop_max = image_size // 2, image_size
    
    curve_p = blur_p = flip_p = grid_p = p
    
    curve_transform = A.OneOf([
        # A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=curve_p, always_apply=always_apply),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=curve_p, always_apply=always_apply),
        A.RandomGamma(gamma_limit=(30, 100), p=curve_p, always_apply=always_apply),
    ], p=curve_p)
    
    blur_transform = A.OneOf([
        A.Blur(blur_limit=5, p=blur_p, always_apply=always_apply),
        A.MedianBlur(blur_limit=3, p=blur_p, always_apply=always_apply),
    ], p=blur_p)
    
    grid_transform = A.OneOf([
        A.ElasticTransform(p=grid_p, alpha=80, sigma=50, alpha_affine=50, always_apply=always_apply),
        A.GridDistortion(p=grid_p, num_steps=9, border_mode=5, always_apply=always_apply),
        A.OpticalDistortion(p=grid_p, distort_limit=0.2, shift_limit=0.2)
    ], p=grid_p)
    
    transform = A.Compose([
        # A.SmallestMaxSize(max_size=image_size),
        # A.RandomCrop(image_size, image_size),
        curve_transform,
        blur_transform,
        A.HorizontalFlip(always_apply=always_apply, p=flip_p),
        A.VerticalFlip(always_apply=always_apply, p=flip_p),
        A.RandomRotate90(always_apply=always_apply, p=flip_p),
        A.Rotate(limit=15, always_apply=always_apply, p=flip_p),
        grid_transform,
        #A.PadIfNeeded(600, 1024, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        # A.RandomSizedCrop(p=crop_p, min_max_height=(crop_min, crop_max), height=image_size, width=image_size, always_apply=always_apply),
        # A.CenterCrop(300, 300, always_apply=always_apply, p=1.0),
        A.GaussNoise(var_limit=(1, 50), p=0.4),
    ], p=1.0)
    
    return transform


def medium_1(image_size, p=0.5):
    return A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.2),
    ], p=p)


def medium_2(image_size, p=1.0):
    cutout_crop = int(0.25*image_size)
    return A.Compose([
        # RandomCrop(input_size) / RandomResizedCrop (0.08, 1)
        A.HorizontalFlip(p=0.5), # vflip
        A.VerticalFlip(p=0.5), # hflip
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.3),
        # shear 10,
        A.RandomBrightnessContrast(brightness_limit=0.125, contrast_limit=0.2, p=0.5), # contrast_limit=0.5
        A.HueSaturationValue(hue_shift_limit=5, p=0.2),
        A.Cutout(num_holes=2, max_h_size=cutout_crop, max_w_size=cutout_crop, p=0.5)
    ], p=p)


def factory(name, image_size, p=0.5, without_norm=False):
    tr_func = globals().get(name, None)
    if tr_func is None:
        raise AttributeError("Transform %s doesn't exist" % (name,))
    
    norm = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]} if not without_norm else None
    
    max_size_tr = A.SmallestMaxSize(max_size=image_size, always_apply=True)
    train_transform = A.Compose([
        max_size_tr,
        tr_func(image_size, p),
        A.RandomCrop(image_size, image_size, always_apply=True),
        ToTensor(normalize=norm),
    ])
    val_transform = A.Compose([
        max_size_tr,
        A.CenterCrop(image_size, image_size, always_apply=True),
        ToTensor(normalize=norm),
    ])
    test_transform = val_transform # TODO: tta (return list)
    
    return train_transform, val_transform, test_transform
