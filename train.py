import datetime
import os

import hydra
import mlflow
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DataDigits, data_load
from model import ModelDigits
from utils import set_seed


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    set_seed(config["seed"])
    df_train = data_load(path="data/df_train.parq")
    df_val = data_load(path="data/df_val.parq")
    train_data = DataDigits(df_train, target_label="target")
    val_data = DataDigits(df_val, target_label="target")
    batch_size = config["batch_size"]
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_data, batch_size=batch_size)
    model = ModelDigits(
        input_dim=len(df_train.drop("target", axis=1).columns),
        out_dim=len(df_train["target"].unique()),
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        batch_norm_flag=config["batch_norm_flag"],
    )
    num_parameters = sum([param.nelement() for param in model.parameters()])
    print(f"Количество параметров в модели: {num_parameters}")
    class_count = df_train["target"].value_counts().values.astype(float)
    class_weights = max(class_count)
    class_weights /= class_count
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights), reduction="mean"
    )

    mlflow.set_tracking_uri(uri=config["mlflow_uri"])
    mlflow.set_experiment(config["experiment_name"])
    run_name = f"run {str(datetime.datetime.now())}"
    git_commit_id = os.popen("git rev-parse head").read()[:-1]
    with mlflow.start_run(
        run_name=run_name,
        tags={"code version": git_commit_id, "author": config["author"]},
    ):
        params = {
            k: v
            for k, v in config.items()
            if k not in ["mlflow_uri", "experiment_name", "author"]
        }
        mlflow.log_params(params)
        for epoch_num in tqdm(range(config["epochs"])):
            train_loss = 0
            val_loss = 0
            train_acc = 0
            val_acc = 0
            train_f1 = 0
            val_f1 = 0
            model.train()
            for data, labels in train_loader:
                output = model(data)
                loss = criterion(output, labels)
                train_loss += loss.item()
                preds_true = output.argmax(dim=1) == labels
                train_acc += preds_true.sum().item() / len(labels)
                train_f1 += f1_score(
                    labels.detach().numpy().astype(int),
                    (output.detach().numpy().argmax(1)).astype(int),
                    average="weighted",
                )
                model.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            for data, labels in val_loader:
                output = model(data)
                loss = criterion(output, labels)
                val_loss += loss.item()
                preds_true = output.argmax(dim=1) == labels
                val_acc += preds_true.sum().item() / len(labels)
                val_f1 += f1_score(
                    labels.detach().numpy().astype(int),
                    (output.detach().numpy().argmax(1)).astype(int),
                    average="weighted",
                )
            train_loss = round(train_loss / len(train_loader), 6)
            mlflow.log_metric("Train Loss", train_loss, epoch_num)
            val_loss = round(val_loss / len(val_loader), 6)
            mlflow.log_metric("Validation Loss", val_loss, epoch_num)
            train_acc = round(train_acc / len(train_loader), 6)
            mlflow.log_metric("Train Accuracy", train_acc, epoch_num)
            val_acc = round(val_acc / len(val_loader), 6)
            mlflow.log_metric("Validation Accuracy", val_acc, epoch_num)
            train_f1 = round(train_f1 / len(train_loader), 6)
            mlflow.log_metric("Train F1-score", train_f1, epoch_num)
            val_f1 = round(val_f1 / len(val_loader), 6)
            mlflow.log_metric("Validation F1-score", val_f1, epoch_num)
    torch.save(model, "model.pth")


if __name__ == "__main__":
    train()
