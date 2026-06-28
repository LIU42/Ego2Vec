import argparse
import os
import json
import tqdm
import torch

import sklearn.metrics as metrics
import torch.nn.functional as functional
import torchvision.transforms as transforms

from lavila.models import models
from lavila.utils import preprocess
from lavila.utils import evaluation
from lavila.utils import distributed

from accelerate import Accelerator
from torch.utils.data import DataLoader

from pytorchvideo.transforms import MoveChannelFront
from pytorchvideo.transforms import Normalize
from pytorchvideo.transforms import ShortSideScale

from datasets import EgocentricDataset
from datasets import DecordDecode
from datasets import TemporalSpatialCrop


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--subset', type=str)

    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--prediction', type=str)
    parser.add_argument('--description', type=str)

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_crops', type=int)
    parser.add_argument('--num_clips', type=int)
    parser.add_argument('--num_workers', type=int)

    parser.add_argument('--clip_length', type=int)
    parser.add_argument('--clip_stride', type=int)

    parser.add_argument('--topk', type=int)

    return parser.parse_args()


def load_models(args):
    checkpoints = torch.load(f'weights/{args.checkpoint}.pth')

    if args.model == 'base':
        model = models.CLIP_OPENAI_TIMESFORMER_BASE(num_frames=args.clip_length)
    elif args.model == 'large':
        model = models.CLIP_OPENAI_TIMESFORMER_LARGE(num_frames=args.clip_length)
    else:
        raise NotImplementedError()

    model.load_state_dict(checkpoints)

    if args.model == 'base':
        tokenizer = preprocess.generate_tokenizer('CLIP_OPENAI_TIMESFORMER_BASE')
    elif args.model == 'large':
        tokenizer = preprocess.generate_tokenizer('CLIP_OPENAI_TIMESFORMER_LARGE')
    else:
        raise NotImplementedError()

    return model, tokenizer


def generate_transform(args):
    video_transform = transforms.Compose([
        DecordDecode(
            args.num_crops,
            args.num_clips,
            args.clip_length,
            args.clip_stride,
        ),
        MoveChannelFront(),
        ShortSideScale(224),
        Normalize(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]),
        TemporalSpatialCrop(
            args.num_crops,
            args.num_clips,
            args.clip_length,
            args.clip_stride,
        ),
    ])

    return video_transform


def load_annotations(dataset):
    annotation_array = []

    for video in dataset.videos:
        label = dataset.annotations[video]
        index = dataset.label2index[label]

        annotation_array.append(index)

    return torch.tensor(annotation_array)


def video_embeddings_name(args):
    return f'video_embeddings_{args.subset}_{args.checkpoint}_{args.num_crops}_{args.num_clips}_{args.clip_length}_{args.clip_stride}'


def label_embeddings_name(args):
    return f'label_embeddings_{args.subset}_{args.checkpoint}'


class ZeroshotPipeline:
    def __init__(self, args):
        self.accelerator = Accelerator()

        self.dataset = EgocentricDataset(args.dataset, args.subset, generate_transform(args))
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

        self.model, self.tokenizer = load_models(args)

        self.checkpoint_name = args.checkpoint
        self.prediction_name = args.prediction
        self.description_name = args.description

        self.topk = args.topk
        self.annotations = load_annotations(self.dataset)

        self.video_embeddings_name = video_embeddings_name(args)
        self.video_embeddings = None

        self.label_embeddings_name = label_embeddings_name(args)
        self.label_embeddings = None

    @property
    def video_embeddings_path(self):
        return f'embeddings/{self.dataset.dataset}/{self.video_embeddings_name}.pth'
    
    @property
    def label_embeddings_path(self):
        return f'embeddings/{self.dataset.dataset}/{self.label_embeddings_name}.pth'
    
    @property
    def description_embeddings_path(self):
        return f'embeddings/{self.dataset.dataset}/description_embeddings_{self.description_name}_{self.checkpoint_name}.pth'
    
    @property
    def is_main_process(self):
        return self.accelerator.is_main_process
    
    def create_embeddings_storage(self):
        if self.is_main_process:
            os.makedirs(f'embeddings/{self.dataset.dataset}', exist_ok=True)

    def encode_video_embeddings(self):
        self.create_embeddings_storage()

        try:
            self.video_embeddings = torch.load(self.video_embeddings_path).cuda()

        except:
            model, dataloader = self.accelerator.prepare(self.model, self.dataloader)

            video_positions = []
            video_embeddings = []

            for positions, frames in tqdm.tqdm(dataloader, 'encoding videos'):
                batch = frames.shape[0]
                count = frames.shape[1]

                frames = frames.flatten(0, 1)
                frames = frames.contiguous()

                embeddings = functional.normalize(distributed.get_model(model).encode_image(frames), dim=1).reshape(batch, count, 256)

                embeddings = embeddings.mean(dim=1)
                embeddings = embeddings.contiguous()

                embeddings = functional.normalize(embeddings, dim=1)

                positions = self.accelerator.gather_for_metrics(positions)
                embeddings = self.accelerator.gather_for_metrics(embeddings)

                video_positions.append(positions)
                video_embeddings.append(embeddings)

            video_positions = torch.cat(video_positions)
            video_embeddings = torch.cat(video_embeddings)

            self.video_embeddings = video_embeddings[video_positions.argsort()]

            if self.is_main_process:
                torch.save(self.video_embeddings.cpu(), self.video_embeddings_path)

    def encode_label_embeddings(self):
        self.create_embeddings_storage()
    
        try:
            self.label_embeddings = torch.load(self.label_embeddings_path).cuda()

        except:
            label_embeddings = []
            
            for label in tqdm.tqdm(self.dataset.labels, 'encoding labels'):
                label_tokens = self.tokenizer([label])

                label_tokens = label_tokens.view(-1, 77).contiguous()
                label_tokens = label_tokens.cuda()

                label_embeddings.append(functional.normalize(distributed.get_model(self.model).encode_text(label_tokens), dim=1))

            self.label_embeddings = torch.cat(label_embeddings)

            if self.is_main_process:
                torch.save(self.label_embeddings.cpu(), self.label_embeddings_path)
    
    def load_descriptions(self):
        with open(f'descriptions/{self.dataset.dataset}/{self.description_name}.json', mode='r') as descriptions:
            return json.load(descriptions)
        
    def encode_description_embeddings(self):
        self.create_embeddings_storage()

        try:
            self.label_embeddings = torch.load(self.description_embeddings_path).cuda()

        except:
            description_embeddings = []
            description_texts = self.load_descriptions()

            for label in tqdm.tqdm(self.dataset.labels, 'encoding descriptions'):
                embeddings = []
                descriptions = description_texts[label]
   
                for description in descriptions:
                    description_tokens = self.tokenizer([description])

                    description_tokens = description_tokens.view(-1, 77).contiguous()
                    description_tokens = description_tokens.cuda()

                    embeddings.append(functional.normalize(distributed.get_model(self.model).encode_text(description_tokens), dim=1).squeeze())

                embeddings = torch.stack(embeddings)

                embeddings = embeddings.mean(dim=0)
                embeddings = embeddings.contiguous()

                description_embeddings.append(functional.normalize(embeddings, dim=0))

            self.label_embeddings = torch.stack(description_embeddings)

            if self.is_main_process:
                torch.save(self.label_embeddings.cpu(), self.description_embeddings_path)
    
    def evaluation_zeroshot(self):
        predictions = self.video_embeddings @ self.label_embeddings.transpose(0, 1)

        if self.dataset.dataset == 'ek100':
            top1_accuracy, top5_accuracy = evaluation.accuracy(predictions, self.annotations, (1, 5))

            print(f'top1 accuracy: {top1_accuracy.item():.3f}')
            print(f'top5 accuracy: {top5_accuracy.item():.3f}')

        else:
            predictions = predictions.argmax(dim=1)
            predictions = predictions.cpu()

            mean_accuracy, top1_accuracy = evaluation.get_mean_accuracy(metrics.confusion_matrix(self.annotations, predictions))

            print(f'mean accuracy: {mean_accuracy:.3f}')
            print(f'top1 accuracy: {top1_accuracy:.3f}')

    def save_prediction_results(self):
        topk_predictions = {}
        _, predictions = torch.topk(self.video_embeddings @ self.label_embeddings.transpose(0, 1), k=self.topk)

        for video in self.dataset.videos:
            index = self.dataset.video2index[video]

            for index in predictions[index]:
                label = self.dataset.index2label[index.item()]

                if video in topk_predictions:
                    topk_predictions[video].append(label)
                else:
                    topk_predictions[video] = [label]

        if self.is_main_process:
            os.makedirs(f'predictions/{self.dataset.dataset}', exist_ok=True)

            with open(f'predictions/{self.dataset.dataset}/{self.prediction_name}.json', mode='w') as predictions:
                json.dump(topk_predictions, predictions)

    @torch.no_grad()
    def zeroshot(self):
        torch.cuda.set_device(self.accelerator.device)

        self.model = self.model.cuda()
        self.model = self.model.eval()

        if self.dataset.dataset == 'ek100':
            self.annotations = self.annotations.cuda()
        else:
            self.annotations = self.annotations.cpu()

        self.encode_video_embeddings()

        if self.is_main_process:
            self.encode_label_embeddings()
            self.evaluation_zeroshot()

            if self.description_name:
                self.encode_description_embeddings()
                self.evaluation_zeroshot()

            if self.prediction_name:
                self.save_prediction_results()


if __name__ == '__main__':
    ZeroshotPipeline(arguments()).zeroshot()
