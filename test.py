import torch
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms
from src.data.dataload import testa_dataset, testb_dataset
from src.models.cnn3D import cnn3D

# 载入模型
model_path = './models/20240112-210346_n_epochs_10.pt'
model = cnn3D(num_classes=3)
model.load_state_dict(torch.load(model_path))
model.eval()

# 载入数据
testa_dataloader = DataLoader(testa_dataset, batch_size=1, shuffle=False)
testb_dataloader = DataLoader(testb_dataset, batch_size=1, shuffle=False)

# 进行预测
predictions = []
name = ['testa', 'testb']
# 先a后b
for i, dataloader in enumerate([testa_dataloader, testb_dataloader]):
    for data, idx in testa_dataloader:
        output = model(data)
        predicted_label = output.argmax(1).item()
        predictions.append({'testa_id': f'{name[i]}_{idx.item()}', 'label': predicted_label})

# 将结果保存到CSV
df = pd.DataFrame(predictions)
df.to_csv('./submission/submission.csv', index=False)
