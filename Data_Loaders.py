import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
        np.random.seed(2024)
        collision_data = self.data[self.data[:, -1] == 1]
        no_collision_data = self.data[self.data[:, -1] == 0]

        sampled_no_collision_data = no_collision_data[
            np.random.choice(len(no_collision_data), len(collision_data), replace=False)
        ]
        balanced_data = np.vstack((collision_data, sampled_no_collision_data))
        np.random.shuffle(balanced_data)

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(balanced_data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return self.normalized_data.shape[0]

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.
        x = np.array(self.normalized_data[idx, 0:6], dtype=np.float32)
        y = np.array(self.normalized_data[idx, 6], dtype=np.float32)

        return {'input': x, 'label': y}

class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        dataset_size = len(self.nav_dataset)
        train_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            self.nav_dataset, [train_size, test_size])
        self.train_loader = data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True)

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
