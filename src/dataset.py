import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FetalPlaneDataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str, train: bool = True, transform = None):

        self.images_dir = images_dir
        self.transform = transform

        # load the CSV
        df = pd.read_csv(csv_path, sep = ";")
        df.columns = df.columns.str.strip()

        # filter train or test
        self.data = df[df["Train"] == (1 if train else 0)].reset_index(drop = True)

        # map class names to integers
        self.classes = ["Fetal abdomen", "Fetal brain", "Fetal femur",
                "Fetal thorax", "Maternal cervix", "Other"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get teh row for this index
        row = self.data.iloc[index]

        # build the full path to the image file
        img_path = os.path.join(self.images_dir, row["Image_name"] + ".png")

        # load the image
        image = Image.open(img_path).convert("L")

        # apply transforms if any
        if self.transform:
            image = self.transform(image)

        # get the label
        label = self.class_to_idx[row["Plane"]]

        return image, label

