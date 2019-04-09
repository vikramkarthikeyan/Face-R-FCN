from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from PIL import Image
import torchvision.transforms as transforms

# https://www.cs.virginia.edu/~vicente/recognition/notebooks/image_processing_lab.html
class IJBADataset(Dataset):

    def __init__(self, data_frame, transform=None):

        self.split_frame = data_frame
        self.transform = transform
        self.pil2tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.split_frame)

    def __getitem__(self, idx):
        try:
            with open(self.split_frame.iloc[idx]['image_location'], 'rb') as f:
                image = Image.open(f)
                image = image.convert('RGB')

        except Exception:
            print("Image not found..", Exception.__traceback__)
            return ([], self.split_frame.iloc[idx]['subject'])

        # if self.transform:
        #     sample = self.transform(sample)

        tensor = self.pil2tensor(image)

        # print(tensor.shape)
