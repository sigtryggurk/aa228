import data_readers as dr
import multift
import numpy as np

from config import Config

if __name__ == "__main__":
  vocab_size = 1000000
  
  with open(Config.MULTIFT_WORDS_FILE) as f:
    multift_words = [w[:-1] for w in f.readlines()][:vocab_size]

  train = dr.get_wic(labels_f=None)
  dev = dr.get_wic(samples_f=Config.WIC_DEV_SAMPLES_FILE, labels_f=None)
  wic_words = set(train.w.tolist()) | set(dev.w.tolist())
  extra_words = list(wic_words - set(multift_words))

  words = multift_words + extra_words
  
  with open(Config.VOCAB_FILE, "w") as f:
    for word in words:
      f.write(word + "\n")

  ft = multift.MultiFastText(basename=Config.MULTIFT_BASE, multi=True, verbose=True, maxn=6)
  emb = np.zeros((len(words), 300, 2))
  for i, word in enumerate(words):
    emb[i] = ft.subword_rep_multi(word)[0]

  np.save(Config.DUAL_SENSE_FILE, emb)



