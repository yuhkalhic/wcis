import argparse
from models import CustomLightningModule, LinearWithLoRA, TimeAxis
from utilities import count_parameters 
from data import get_dataset, tokenization, setup_dataloaders
import time
from functools import partial

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from transformers import AutoModelForSequenceClassification
import torch

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true'):
        return True
    elif v.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def train_model():
    parser = argparse.ArgumentParser(description='LoRA parameters configuration')
    parser.add_argument('--lora_r', type=int, default=8, help='Rank for LoRA layers')
    parser.add_argument('--lora_alpha', type=int, default=16, help='Alpha for LoRA layers')
    parser.add_argument('--lora_query', type=str2bool, default=True, help='Apply LoRA to query')
    parser.add_argument('--lora_key', type=str2bool, default=False, help='Apply LoRA to key')
    parser.add_argument('--lora_value', type=str2bool, default=True, help='Apply LoRA to value')
    parser.add_argument('--lora_projection', type=str2bool, default=False, help='Apply LoRA to projection')
    parser.add_argument('--lora_mlp', type=str2bool, default=False, help='Apply LoRA to MLP')
    parser.add_argument('--lora_head', type=str2bool, default=False, help='Apply LoRA to head')
    parser.add_argument('--device', type=int, default=0, help='Specify GPU device index')
    parser.add_argument('--verbose', type=str2bool, default=True, help='Enable/disable progress bars')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Please switch to a GPU machine before running this code.")
        quit()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_axis = TimeAxis(device=device)

    df_train, df_val, df_test = get_dataset()
    imdb_tokenized = tokenization()
    train_loader, val_loader, test_loader = setup_dataloaders(imdb_tokenized)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2
    ).to(device)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    assign_lora = partial(LinearWithLoRA, rank=args.lora_r, alpha=args.lora_alpha, time_axis=time_axis, device=device)

    layer_keys = {
        "q_lin": "query",
        "k_lin": "key",
        "v_lin": "value",
        "out_lin": "output",
        "lin1": "ffn1",
        "lin2": "ffn2",
        "pre_classifier": "pre_cls",
        "classifier": "cls"
    }

    for layer in model.distilbert.transformer.layer:
        if args.lora_query:
            layer.attention.q_lin = assign_lora(layer.attention.q_lin, layer_key=layer_keys["q_lin"])
        if args.lora_key:
            layer.attention.k_lin = assign_lora(layer.attention.k_lin, layer_key=layer_keys["k_lin"])
        if args.lora_value:
            layer.attention.v_lin = assign_lora(layer.attention.v_lin, layer_key=layer_keys["v_lin"])
        if args.lora_projection:
            layer.attention.out_lin = assign_lora(layer.attention.out_lin, layer_key=layer_keys["out_lin"])
        if args.lora_mlp:
            layer.ffn.lin1 = assign_lora(layer.ffn.lin1, layer_key=layer_keys["lin1"])
            layer.ffn.lin2 = assign_lora(layer.ffn.lin2, layer_key=layer_keys["lin2"])
    if args.lora_head:
        model.pre_classifier = assign_lora(model.pre_classifier, layer_key=layer_keys["pre_classifier"])
        model.classifier = assign_lora(model.classifier, layer_key=layer_keys["classifier"])

    print("Total number of trainable parameters:", count_parameters(model))


    lightning_model = CustomLightningModule(model)
    callbacks = [
        ModelCheckpoint(
            save_top_k=1, mode="max", monitor="val_acc"
        ) # save top 1 model
    ]
    logger = CSVLogger(save_dir="logs/", name=f"my-model-{args.device}")

    trainer = L.Trainer(
        max_epochs=3,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accelerator="gpu",
        precision="16-mixed",
        devices=[int(args.device)],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=args.verbose
    )

    start = time.time()

    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

    train_acc = trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best", verbose=False)
    val_acc = trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best", verbose=False)
    test_acc = trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best", verbose=False)

    # Print all argparse settings
    print("------------------------------------------------")
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    train_acc = trainer.test(lightning_model, dataloaders=train_loader, ckpt_path="best", verbose=False)
    val_acc = trainer.test(lightning_model, dataloaders=val_loader, ckpt_path="best", verbose=False)
    test_acc = trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best", verbose=False)

    # Print settings and results
    with open("results.txt", "a") as f:
        s = "------------------------------------------------"
        print(s), f.write(s+"\n")
        for arg in vars(args):
            s = f'{arg}: {getattr(args, arg)}'
            print(s), f.write(s+"\n")

        s = f"Train acc: {train_acc[0]['accuracy']*100:2.2f}%"
        print(s), f.write(s+"\n")
        s = f"Val acc:   {val_acc[0]['accuracy']*100:2.2f}%"
        print(s), f.write(s+"\n")
        s = f"Test acc:  {test_acc[0]['accuracy']*100:2.2f}%"
        print(s), f.write(s+"\n")
        s = "------------------------------------------------"
        print(s), f.write(s+"\n")

if __name__ == "__main__":
    train_model()