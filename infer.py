import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import DataDigits, data_load


def infer():
    df_test = data_load(path="data/df_test.parq")
    test_data = DataDigits(df_test, target_label="target")
    with open("config/config.yaml", "r") as f:
        batch_size = yaml.safe_load(f)["batch_size"]
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model = torch.load("model.pth")
    model.eval()
    result = []
    for data, _ in tqdm(test_loader):
        output = model(data)
        result.extend((output.detach().numpy().argmax(1)).astype(int))
    df_test["predict_target"] = result
    df_test.to_csv("result.csv", index=False)


if __name__ == "__main__":
    infer()
