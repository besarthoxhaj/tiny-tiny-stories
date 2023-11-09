import sentencepiece as spm


class TinyTokenizer:
  def __init__(self, prefix='tiny_piece'):
    self.prefix = prefix
    self.sp = spm.SentencePieceProcessor()

  def encode(self, txt):
    return self.sp.encode_as_ids(txt)

  def decode(self, ids):
    return self.sp.decode_ids(ids)

  def train(self, path):
    spm.SentencePieceTrainer.Train(
      input=path,
      model_type='bpe',
      model_prefix=self.prefix,
      vocab_size=10000,
      pad_id=0,
      unk_id=1,
      bos_id=2,
      eos_id=3,
      pad_piece='[PAD]',
      unk_piece='[UNK]',
      bos_piece='[BOS]',
      eos_piece='[EOS]'
    )

    return self

  def load(self):
    self.sp.load(f'./{self.prefix}.model')
    return self

  def vocab_size(self):
    return self.sp.get_piece_size()


if __name__ == '__main__':
  tknz = TinyTokenizer()
  tknz.train('./tiny_stories_all.txt').load()
  # tknz.load()
  print("tknz.vocab_size()", tknz.vocab_size())
  print('tknz.sp.bos_id()', tknz.sp.bos_id())
  print('tknz.sp.pad_id()', tknz.sp.pad_id())
  print('tknz.sp.eos_id()', tknz.sp.eos_id())
  print('tknz.sp.unk_id()', tknz.sp.unk_id())

  ids_foo = tknz.encode('hello my name is Bes')
  ids_bar = tknz.encode('ciao il mio nome è Bes')
  ids_zoo = tknz.encode('emma')
  print('ids_foo', ids_foo)
  print('ids_bar', ids_bar)
  print('ids_zoo', ids_zoo)
  txt_foo = tknz.decode(ids_foo)
  txt_bar = tknz.decode(ids_bar)
  txt_zoo = tknz.decode(ids_zoo)
  print('txt_foo', txt_foo)
  print('txt_bar', txt_bar)
  print('txt_zoo', txt_zoo)
  for id in range(4): print(id, tknz.sp.id_to_piece(id), tknz.sp.is_control(id))