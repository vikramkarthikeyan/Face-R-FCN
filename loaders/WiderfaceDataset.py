from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import scipy.io

# https://www.cs.virginia.edu/~vicente/recognition/notebooks/image_processing_lab.html
class WiderFaceDataset(Dataset):

    def __init__(self, image_path, metadata_path, transform=None):
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.transform = transform
        self.pil2tensor = transforms.ToTensor()
        
        self.convert_to_image_list(self.metadata_path)
    
    def convert_to_image_list(self, path):
        self.f = scipy.io.loadmat(path)
        self.event_list = self.f.get('event_list')
        self.file_list = self.f.get('file_list')
        self.face_bbx_list = self.f.get('face_bbx_list')


        data_frame = []

        for idx, event in enumerate(self.event_list):
            event = event[0][0]

            for file_idx, file_path in enumerate(self.file_list[idx]):
                file_name = file_path[0][0][0] + '.jpg'
                
                bounding_boxes = self.face_bbx_list[idx][file_idx][0][0]
                
                for bbx_idx, bounding_box in enumerate(bounding_boxes):
                    print(bounding_box)

                data_frame.append((event, file_name, self.face_bbx_list[idx][file_idx]))
            
            print("\n")
            
            print("separator...\n\n\n")
            
            # for bounding_boxes in self.face_bbx_list[idx]:
            #     print(bounding_boxes)
            #     print(data_frame[idx])
            #     data_frame[idx].append(bounding_boxes)
        
        print(data_frame)
        


            


    def __len__(self):
        return len(self.image_location_list)

    def __getitem__(self, idx):
        try:
            with open(self.image_location_list.iloc[idx]['image_location'], 'rb') as f:
                image = Image.open(f)
                image = image.convert('RGB')

        except Exception:
            print("Image not found..", Exception.__traceback__)
            return ([], self.split_frame.iloc[idx]['subject'])

        # if self.transform:
        #     sample = self.transform(sample)

        tensor = self.pil2tensor(image)

        # print(tensor.shape)
