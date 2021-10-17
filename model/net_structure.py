import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# text->sections classifier
class Net_section_classifier(pl.LightningModule):
  def __init__(self, n_feats: int, n_classes: int):
    super().__init__()
    self.fc1 = nn.utils.weight_norm(nn.Linear(n_feats, 128))
    self.dropout = nn.Dropout(0.4)
    self.fc2 = nn.utils.weight_norm(nn.Linear(128, n_classes))
    self.criterion = nn.BCELoss()

  def forward(self, inputs, labels=None):
    h = self.fc1(inputs)
    h = self.dropout(h)
    h = F.relu(h)
    h = self.dropout(h)
    h = self.fc2(h)
    outputs = torch.sigmoid(h)
    loss = 0
    if labels is not None:
      loss = self.criterion(outputs, labels)
    return loss, outputs

# config
dropout = [
           0.2525876416930929,
           0.3943931037931156,
           0.47613467828542394,
           0.41803846653078797,
           0.3432586380577326,
           ]
           
hidden_size = [200, 800, 1000, 200]

config = {
    'activation': nn.SiLU,
    'dropout': dropout,
    'hidden_size': hidden_size,
    'lr': 0.00028597754641582484,
    'num_layer': 4,
    'optimizer': 'Adam'
    }

# text->subclasses classifier
class Net_subclass_classifier(pl.LightningModule):
  def __init__(self, n_feats, n_classes, config):
    super(Net_subclass_classifier, self).__init__()
    self.input_shape = n_feats
    self.n_classes = n_classes
    self.num_layer = config['num_layer']
    self.dropout = config['dropout']
    self.hidden_size = config['hidden_size']
    self.activation = config['activation']
    self.criterion = nn.BCELoss()
    self.optimizer_name = config['optimizer']
    self.lr = config['lr']
  
    # define layers
    self.layers = []
    for i in range(self.num_layer):
        self.out_shape = self.hidden_size[i]
        self.layers.append(nn.Dropout(self.dropout[i]))
        self.layers.append(nn.Linear(self.input_shape, self.out_shape))
        self.layers.append(nn.BatchNorm1d(self.out_shape))
        self.layers.append(self.activation())
        # update input shape
        self.input_shape = self.out_shape
    # define the final layer
    self.layers.append(nn.Dropout(self.dropout[-1]))
    self.layers.append(nn.Linear(self.input_shape, self.n_classes))
    self.layers.append(nn.Sigmoid())

    self.model = torch.nn.Sequential(*self.layers) # *はlistの中身をばらばらで渡す

  def forward(self, inputs, labels=None):
    outputs = self.model(inputs)
    loss = 0
    if labels is not None:
      loss = self.criterion(outputs, labels)
    return loss, outputs

  def training_step(self, batch, batch_idx):
    inputs, labels = batch
    loss, outputs = self(inputs, labels)
    return {'loss': loss, 'predictions': outputs, 'labels': labels}

  def validation_step(self, batch, batch_idx):
    inputs, labels = batch
    loss, outputs = self(inputs, labels)
    self.log('val_loss', loss, on_step=False, on_epoch=True)
    return {'loss': loss, 'predictions': outputs, 'labels': labels}

  def configure_optimizers(self):
    optimizer = getattr(optim, self.optimizer_name)(self.model.parameters(), lr=self.lr)
    return optimizer