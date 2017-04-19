import torch.utils.data
import torchvision.transforms as transforms
from data.base_data_loader import BaseDataLoader
from data.image_folder import ImageFolder
from builtins import object


class PairedData(object):

    def __init__(self,data_loader_A_images, data_loader_A_labels, data_loader_B_images, data_loader_B_labels, data_loader_AB_images_1 , data_loader_AB_images_2):
        self.data_loader_A_images = data_loader_A_images
        self.data_loader_B_images = data_loader_B_images
        self.data_loader_A_labels = data_loader_A_labels
        self.data_loader_B_labels = data_loader_B_labels
        self.data_loader_AB_images_1 = data_loader_AB_images_1
        self.data_loader_AB_images_2 = data_loader_AB_images_2

    def __iter__(self):
        self.data_loader_A_images_iter = iter(data_loader_A_images)
        self.data_loader_B_images_iter = iter(data_loader_B_images)
        self.data_loader_A_labels_iter = iter(data_loader_A_labels)
        self.data_loader_B_labels_iter = iter(data_loader_B_labels)
        self.data_loader_AB_images_1_iter = iter(data_loader_AB_images_1)
        self.data_loader_AB_images_2_iter = iter(data_loader_AB_images_2)
        return self

    def __next__(self):
        A_image, A_image_paths= next(self.data_loader_A_images_iter)
        B_image, B_image_paths=next(self.data_loader_B_images_iter)

        A_label, A_label_paths=next(self.data_loader_A_labels_iter)
        B_label, B_label_paths=   next(self.data_loader_B_labels_iter)

        AB_image_1, AB_image_1_paths=next(self.data_loader_AB_images_1_iter)
        AB_image_2, AB_image_1_paths=   next(self.data_loader_AB_images_2_iter)


        return {'A_image': A_image, 'A_image_paths': A_image_paths,
                'B_image': B_image, 'B_image_paths': B_image_paths, 'A_label': A_label, 'A_label_paths': A_label_paths,
                'B_label': B_label, 'B_label_paths': B_label_paths,'AB_image_1': AB_image_1, 'AB_image_1_paths': AB_image_1_paths,
                'AB_image_2': AB_image_2, 'AB_image_1_paths': AB_image_1_paths}


class CombinedDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transform = transforms.Compose([
                                       transforms.Scale(opt.loadSize),
                                       transforms.CenterCrop(opt.fineSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))])

        # Dataset A
        domain_A_images = ImageFolder(root=opt.dataroot + '/' + opt.domain_A + '/images',
                                transform=transform, return_paths=True)

        # Dataset A
        domain_A_labels = ImageFolder(root=opt.dataroot + '/' + opt.domain_A + '/labels',
                                transform=transform, return_paths=True)


        domain_B_images= ImageFolder(root=opt.dataroot + '/' + opt.domain_B + '/images',
                                transform=transform, return_paths=True)

        # Dataset A
        domain_B_labels = ImageFolder(root=opt.dataroot + '/' + opt.domain_B + '/labels',
                                transform=transform, return_paths=True)


        # Dataset AB
        domain_AB_images_1 = ImageFolder(root=opt.dataroot + '/' + opt.domain_A + '/images',
                                transform=transform, return_paths=True,sort=False)

        # Dataset AB
        domain_AB_images_2 = ImageFolder(root=opt.dataroot + '/' + opt.domain_B + '/images',
                                transform=transform, return_paths=True,sort=False)

        data_loader_A_images = torch.utils.data.DataLoader(
            domain_A_images,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))

        data_loader_A_labels = torch.utils.data.DataLoader(
            domain_A_labels,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))

        data_loader_B_images = torch.utils.data.DataLoader(
            domain_B_images,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))

        data_loader_B_labels = torch.utils.data.DataLoader(
            domain_B_labels,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))

        data_loader_AB_images_1 = torch.utils.data.DataLoader(
            domain_AB_images_1 ,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))

        data_loader_AB_images_2  = torch.utils.data.DataLoader(
            domain_AB_images_2 ,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))
        

        self.domain_A_images = domain_A_images
        self.domain_A_labels = domain_A_labels
        self.domain_B_images = domain_A_images
        self.domain_B_labels = domain_A_labels
        self.domain_AB_images_1 = domain_AB_images_1
        self.domain_AB_images_2 = domain_AB_images_2 


        self.paired_data = PairedData(data_loader_A_images, data_loader_A_labels, data_loader_B_images, data_loader_B_labels, data_loader_AB_images_1 , data_loader_AB_images_2)

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return len(self.dataset_A)
