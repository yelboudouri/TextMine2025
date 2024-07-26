import argparse
import os
from multiprocessing import cpu_count
from pathlib import Path

import torch
import wandb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.trainers import BpeTrainer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig

from dataset import TextMineDataset, prepare_data, ENTITY_CLASSES, RELATION_CLASSES
from utils import cycle, CosineAnnealingWithWarmRestartsLR


WANDB_API_KEY = os.getenv("WANDB_API_KEY")


class Trainer:

    def __init__(self,
                 output_dir: Path,
                 train_dataloader,
                 eval_dataloader,
                 model,
                 device,
                 training_steps: int = 1000000,
                 lr: float = 1e-6,
                 use_amp: bool = True,
                 checkpoint: Path = None, ):
        self.output_dir = output_dir
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.model = model
        self.device = device
        self.training_steps = training_steps
        self.use_amp = use_amp

        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingWithWarmRestartsLR(self.optimizer, warmup_steps=128, cycle_steps=1024,
                                                           max_lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        if checkpoint is not None:
            self.load(checkpoint)

        wandb.login(key=WANDB_API_KEY)
        wandb.init(project="TextMine", resume="allow", id="run_22")

    def train(self):
        self.model.train()

        eval_loss = None
        eval_accuracy = None
        macro_f1 = None

        pbar = tqdm(unit="batch", total=self.training_steps, desc="Training: ")
        for step in range(self.training_steps):
            input_ids, attention_mask, token_type_ids, targets = next(self.train_dataloader)

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            targets = targets.to(self.device)

            with torch.autocast(device_type=self.device, enabled=self.use_amp):
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                    labels=targets)
            loss = output.loss

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()

            if step % 4096 == 0 and step > 0:
                eval_loss, eval_accuracy, macro_f1 = self.val()
                self.save()

                wandb.log({
                    'eval_loss': eval_loss,
                    'eval_accuracy': eval_accuracy,
                    'macro_f1': macro_f1
                })

            # Update progress bar
            pbar.set_postfix({
                'train_loss': loss.item(),
                'eval_loss': eval_loss,
                'eval_accuracy': eval_accuracy,
                'macro_f1': macro_f1
            })
            pbar.update(1)

            wandb.log({
                'train_loss': loss.item()
            })

    @torch.no_grad()
    def val(self):
        self.model.eval()

        losses = []
        predicted_labels = []
        true_labels = []

        for data_point in tqdm(self.eval_dataloader):
            input_ids, attention_mask, token_type_ids, targets = data_point

            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            targets = targets.to(self.device)

            output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                labels=targets)
            loss = output.loss
            logits = output.logits
            predictions = torch.argmax(logits, dim=1)

            losses.append(loss.item())
            predicted_labels.extend(predictions.tolist())
            true_labels.extend(targets.tolist())

        self.model.train()

        average_loss = sum(losses) / len(losses)
        accuracy = accuracy_score(true_labels, predicted_labels)
        macro_f1 = f1_score(true_labels, predicted_labels, average='macro')

        report = classification_report(true_labels, predicted_labels)
        print(report)

        return average_loss, accuracy, macro_f1

    def save(self):
        dt = {
            'model': self.model.state_dict(),
            'opt': self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        torch.save(dt, str(self.output_dir / 'last_model.pt'))

    def load(self, checkpoint):
        print(f"Loaded checkpoint from: {checkpoint}")
        dt = torch.load(args.checkpoint, map_location=self.device)

        self.model.load_state_dict(dt['model'])
        self.optimizer.load_state_dict(dt['opt'])
        self.scaler.load_state_dict(dt['scaler'])
        self.scheduler.load_state_dict(dt['scheduler'])


def train_tokenizer(vocab_size, data, output_directory):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation()])
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]"], min_frequency=8)
    tokenizer.train_from_iterator(data, trainer=trainer)

    tokenizer.save(str(output_directory / 'tokenizer.json'))
    print(f"Tokenizer saved to: {output_directory / 'tokenizer.json'}")

    return tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Dataset Processing')

    parser.add_argument('--dataset-path', type=str, default='data',
                        help='Path to the dataset directory (default: data)')
    parser.add_argument('--output-directory', type=str, default='output',
                        help='Directory to save the output (default: output)')
    parser.add_argument('--vocab-size', type=int, default=4096 * 4,
                        help='Vocabulary size (default: 4096 * 4)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing (default: 32)')
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint file for resuming or fine-tuning training.",
                        )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Load the dataset
    train_set, eval_set, max_length = prepare_data(f'{args.dataset_path}/train.csv', tokenizer)

    train_ds = TextMineDataset(train_set, tokenizer, max_length, upsample=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=cpu_count(), drop_last=True,
                          pin_memory=True, shuffle=True)
    train_dl = cycle(train_dl)

    eval_ds = TextMineDataset(eval_set, tokenizer, max_length)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch_size, num_workers=cpu_count(), pin_memory=True)

    # Initialize the model
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer._convert_token_to_id('[PAD]'),
        type_vocab_size=len(ENTITY_CLASSES) * 2,
        num_labels=len(RELATION_CLASSES)

    )
    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", config=config,
                                                          ignore_mismatched_sizes=True)
    model.to(device)

    # Train the model
    Trainer(
        output_dir=output_directory,
        checkpoint=args.checkpoint,
        train_dataloader=train_dl,
        eval_dataloader=eval_dl,
        model=model,
        device=device
    ).train()
