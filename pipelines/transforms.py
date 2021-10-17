from torchvision import transforms

def augmentation_transform(config : dict):
	transform = transforms.Compose([
		transforms.RandomAffine(
			degrees=config["rotation_augmentations"],
			translate=(0,0)),
		transforms.RandomHorizontalFlip(),
		transforms.Resize((config["image_dimention"], config["image_dimention"]), Image.BICUBIC),
		#transforms.RandomCrop(res, padding=4),
		#transforms.CenterCrop(res),
		transforms.ToTensor(),
		#transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
		transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
	])

	return transform

submission_transform = transforms.Compose([
	transforms.Resize((config["image_dimention"], config["image_dimention"]), Image.BICUBIC),
	transforms.ToTensor(),
	transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
