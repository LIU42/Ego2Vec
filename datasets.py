import decord
import json
import torch
import torch.nn.functional as functional

from torch.nn import Module
from torch.utils.data import Dataset


def load_metadata(dataset, subset):
    with open(f'datasets/{dataset}/metadata_{subset}.json', mode='r') as dataset_metadata:
        return json.load(dataset_metadata)
    

def decode_video(reader, indices):
    return torch.as_tensor(reader.get_batch(indices).asnumpy()).float()


def padding(frames, padding_length=16, padding_value=0):
    length = frames.shape[0]

    if length < padding_length:
        padding_sequence = [0, 0, 0, 0, 0, 0, 0, padding_length - length]
    else:
        padding_sequence = [0, 0, 0, 0, 0, 0, 0, 0]

    return functional.pad(frames, padding_sequence, value=padding_value)


def spatial_crop(frames, cropping_size=224, position=1):
    if position == 1:
        x1 = (frames.shape[4] - cropping_size) // 2
        y1 = (frames.shape[3] - cropping_size) // 2

        x2 = x1 + cropping_size
        y2 = y1 + cropping_size

        return frames[:, :, :, y1:y2, x1:x2]
    
    elif position == 2:
        return frames[:, :, :, :cropping_size, :cropping_size]
    
    elif position == 3:
        x2 = frames.shape[4]
        y2 = frames.shape[3]

        x1 = x2 - cropping_size
        y1 = y2 - cropping_size

        return frames[:, :, :, y1:y2, x1:x2]
    
    else:
        raise NotImplementedError()
        

class DecordDecode(Module):
    def __init__(self, crops=3, clips=10, length=16, stride=2):
        super().__init__()

        self.crops = crops
        self.clips = clips
        
        self.length = length
        self.stride = stride

    def sparse_sampling_decode(self, reader):
        sampling_center = []
        sampling_length = len(reader) / self.length

        for index in range(self.length):
            upper_bound = int(sampling_length * index + sampling_length)
            lower_bound = int(sampling_length * index)

            sampling_center.append((upper_bound + lower_bound) // 2)

        return decode_video(reader, sampling_center)
    
    def dense_sampling_decode(self, reader):
        length = len(reader)

        if length > self.sequence_length:
            sampling_indices = []
            sampling_results = torch.linspace(0, length - self.sequence_length, self.clips).long()

            for position in sampling_results:
                upper_bound = position.item() + self.sequence_length
                lower_bound = position.item()

                sampling_indices.extend(range(lower_bound, upper_bound, self.stride))

            return decode_video(reader, sampling_indices)
        
        else:
            return padding(decode_video(reader, range(0, length, self.stride)), self.length).repeat(self.clips, 1, 1, 1)

    def forward(self, reader):
        if self.clips == 1:
            return self.sparse_sampling_decode(reader)
        else:
            return self.dense_sampling_decode(reader)
        
    @property
    def sequence_length(self):
        return self.length * self.stride
        

class TemporalSpatialCrop(Module):
    def __init__(self, crops=3, clips=10, length=16, stride=2):
        super().__init__()

        self.crops = crops
        self.clips = clips

        self.length = length
        self.stride = stride

    def forward(self, frames):
        channels, _, height, width = frames.shape

        if self.clips == 1:
            frames = frames.unsqueeze(dim=0)
        else:
            frames = frames.reshape(channels, self.clips, self.length, height, width).transpose(0, 1)

        cropping_size = min(width, height)

        if self.crops == 1:
            return spatial_crop(frames, cropping_size)
    
        elif self.crops == 3:
            return torch.cat([
                spatial_crop(frames, cropping_size, position=1),
                spatial_crop(frames, cropping_size, position=2),
                spatial_crop(frames, cropping_size, position=3),
            ])

        else:
            raise NotImplementedError()


class EgocentricDataset(Dataset):
    def __init__(self, dataset, subset, transform=None):
        super().__init__()
        dataset_metadata = load_metadata(dataset, subset)

        self.dataset = dataset
        self.transform = transform

        self.videos = dataset_metadata['videos']
        self.labels = dataset_metadata['labels']
        self.annotations = dataset_metadata['annotations']

        self.label2index = {}
        self.index2label = {}

        self.video2index = {}
        self.index2video = {}

        for index, label in enumerate(self.labels):
            self.label2index[label] = index
            self.index2label[index] = label

        for index, video in enumerate(self.videos):
            self.video2index[video] = index
            self.index2video[index] = video

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        reader = decord.VideoReader(f'datasets/{self.dataset}/videos/{self.index2video[index]}.mp4')

        if self.transform:
            return index, self.transform(reader)
        else:
            raise NotImplementedError()
