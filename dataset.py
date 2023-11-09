#%%
import torch
import tokenizer
import datasets


class TinyDataset(torch.utils.data.Dataset):
  def __init__(self):
    self.ds = datasets.load_dataset("roneneldan/TinyStories")
    self.tk = tokenizer.TinyTokenizer()
    self.tk.load()

  def __len__(self):
    return len(self.ds['train'])

  def __getitem__(self, idx):
    row = self.ds['train'][idx]['text']
    input = [self.tk.sp.bos_id()] + self.tk.encode(row)
    label = (self.tk.encode(row)) + [self.tk.sp.eos_id()]
    return { 'input': torch.tensor(input), 'label': torch.tensor(label) }

  def collate_fn(self, batch):
    input_pad = torch.nn.utils.rnn.pad_sequence([item['input'] for item in batch], batch_first=True, padding_value=0)
    label_pad = torch.nn.utils.rnn.pad_sequence([item['label'] for item in batch], batch_first=True, padding_value=0)
    return { 'input': input_pad, 'label': label_pad }

#%%
if __name__ == '__main__':
  ds = TinyDataset()
  print('ds.ds', ds.ds)
  print('len(ds)', len(ds))
  print('ds[362]', ds[362])
# %%
