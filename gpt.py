import torch


EMBD = 256
HEAD = 4
DROP = 0.1
SQNZ = 1024
VOCB = 10000


class Attention(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.qkv_proj = torch.nn.Linear(EMBD, EMBD * 3)
    self.out_proj = torch.nn.Linear(EMBD, EMBD)
    self.register_buffer('mask', torch.tril(torch.ones(SQNZ, SQNZ).view(1, 1, SQNZ, SQNZ)))

  def forward(self, x):
    B, S, E = x.shape
    EMBD_HEAD = int(EMBD / HEAD)

    qry, key, val = self.qkv_proj(x).split(EMBD, dim=-1)
    qry = qry.reshape(B, S, HEAD, EMBD_HEAD).transpose(1, 2)
    key = key.reshape(B, S, HEAD, EMBD_HEAD).transpose(1, 2)
    val = val.reshape(B, S, HEAD, EMBD_HEAD).transpose(1, 2)

    msk = self.mask[:, :, :S, :S] == 0
    att = qry @ key.transpose(-1, -2) / torch.sqrt(torch.tensor(EMBD_HEAD))
    att = att.masked_fill(msk, float('-inf'))
    att = torch.nn.functional.softmax(att, dim=-1)
    out = (att @ val).transpose(1, 2).reshape(B, S, E)
    return self.out_proj(out)


class FeedForward(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.c_fc   = torch.nn.Linear(EMBD, EMBD * 4)
    self.relu   = torch.nn.ReLU()
    self.c_proj = torch.nn.Linear(EMBD * 4, EMBD)
    self.drop   = torch.nn.Dropout(DROP)

  def forward(self, x):
    x = self.c_fc(x)
    x = self.relu(x)
    x = self.c_proj(x)
    x = self.drop(x)
    return x


class Block(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.ln_1 = torch.nn.LayerNorm(EMBD)
    self.attn = Attention()
    self.ln_2 = torch.nn.LayerNorm(EMBD)
    self.ffww = FeedForward()

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.ffww(self.ln_2(x))
    return x


class GPT(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.tok_emb = torch.nn.Embedding(VOCB, EMBD)
    self.pos_emb = torch.nn.Embedding(SQNZ, EMBD)
    self.drop    = torch.nn.Dropout(DROP)
    self.blocks  = torch.nn.ModuleList([Block() for _ in range(4)])
    self.norm    = torch.nn.LayerNorm(EMBD)
    self.vocab   = torch.nn.Linear(EMBD, VOCB)

  def forward(self, x):
    tok_emb = self.tok_emb(x)
    pos_emb = self.pos_emb(torch.arange(x.size(1)))
    x = self.drop(tok_emb + pos_emb)
    for block in self.blocks: x = block(x)
    x = self.norm(x)
    x = self.vocab(x)
    return x

  def num_params(self):
    gpt_params = sum(p.numel() for p in self.parameters())
    emb_params = self.tok_emb.weight.numel()
    print(f"Total Parameters: {gpt_params} | Embedding: {emb_params}")
    return { 'gpt_params': gpt_params, 'emb_params': emb_params }

  def generate(self, x, temp=1.0, num=10):
    self.eval()
    for _ in range(num):
      with torch.no_grad():
        logits = self(x)
        logits = logits[:, -1, :] / temp
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next = torch.multinomial(probs, num_samples=1)
        if next.item() == 1: break
        x = torch.cat([x, next], dim=1)
    self.train()
    return x