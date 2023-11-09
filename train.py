import torch
import gpt
import dataset
import tokenizer
import random


is_cuda = torch.cuda.is_available()
device = "cuda:0" if is_cuda else "cpu"


random.seed(42)
torch.manual_seed(42)
myGPT = gpt.GPT().to(device)
myGPT.num_params()


tk = (tokenizer.TinyTokenizer()).load()
ds = dataset.TinyDataset()
dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, collate_fn=ds.collate_fn)
opt = torch.optim.Adam(myGPT.parameters(), lr=0.0001)


for epoch in range(5):
  org = "Hello my name is Bes and I work in the field of AI."
  src = torch.tensor([tk.encode(org)]).to(device)
  trs = myGPT.generate(src)
  print(f"{org} - {tk.decode(trs.tolist()[0])}")

  for idx, batch in enumerate(dl):
    x = batch['input'].to(device)
    y = batch['label'].to(device)
    p = myGPT(x)

    p = p.view(-1, p.size(-1))
    y = y.view(-1)
    l = torch.nn.functional.cross_entropy(p, y, ignore_index=0)
    if idx % 1000 == 0: print(f"Loss: {l.item():.4f}")
    if idx % 5000 == 0: torch.save(myGPT.state_dict(), f"weights_{epoch}_{idx}.pt")
    l.backward()
    opt.step()
    opt.zero_grad()