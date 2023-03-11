import torchvision.transforms as transforms


def create_transform(img_size: int, channels: int,
                     has_random_horizontal_flip: bool = False) -> transforms.Compose:
    pipeline = [transforms.Resize([img_size, img_size])]

    if has_random_horizontal_flip:
        pipeline.append(transforms.RandomHorizontalFlip())

    if channels == 1:
        pipeline.append(transforms.Grayscale())
    
    pipeline.extend([transforms.ToTensor(),
                     transforms.Normalize([0.5] * channels, [0.5] * channels)])

    return transforms.Compose(pipeline)
