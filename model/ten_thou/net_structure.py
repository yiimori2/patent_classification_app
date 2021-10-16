import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# text->subclasses classifier
class Net_subclass_classifier(pl.LightningModule):
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
