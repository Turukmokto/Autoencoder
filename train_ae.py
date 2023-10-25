import torch
import torch.nn as nn
from tqdm import tqdm
from autoencoder import Autoencoder
from utils import make_dataloader
from utils import B_COEF
from utils import device
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    BATCH_SIZE = 16

    dataset_path = 'dataset'

    image_datasets, dataloader = make_dataloader(dataset_path, BATCH_SIZE)

    for B in B_COEF:

        model = Autoencoder(B=B).to(device)
        EPOCHS = 3
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                               factor=0.5, patience=5,
                                                               verbose=True, threshold=0.0001,
                                                               threshold_mode='rel', cooldown=0,
                                                               min_lr=0, eps=1e-08)
        for epoch in range(EPOCHS):
            for phase in ['Train', 'Test']:
                if phase == 'Train':
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                for inputs in tqdm(dataloader[phase]):
                    inputs = inputs.to(device)

                    predicted = model(inputs)

                    loss = criterion(predicted, inputs)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.detach().cpu().numpy()
                epoch_loss = running_loss / len(image_datasets[phase])
                print(f'Epoch{epoch + 1} {phase} loss: {epoch_loss}')
                torch.save(model.encoder.state_dict(), f'new_weights/{B}/encode_epoch{epoch + 1}.pt')
                torch.save(model.decoder.state_dict(), f'new_weights/{B}/decode_epoch{epoch + 1}.pt')
                torch.save(model.state_dict(), f'new_weights/{B}/ae_epoch{epoch + 1}.pt')
                scheduler.step(epoch_loss)
                plt.imshow(np.concatenate([inputs[0].detach().cpu().numpy().transpose(1, 2, 0),
                                           predicted[0].detach().cpu().numpy().transpose(1, 2, 0)], axis=1))
                plt.show()
