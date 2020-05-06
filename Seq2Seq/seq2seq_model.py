#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torchtext
import torch
import torch.nn as nn
from torchtext.data import Field, Example
import argparse
import spacy
from nmtmodel import NMTModel, Encoder, Decoder
import json
def reader(path):
    with open(path) as fp:
        for line in fp:
            yield line


src_field = Field(init_token= "<bos>" , eos_token="<eos>",use_vocab=True,tokenize='spacy' , batch_first= True)
tgt_filed = Field(init_token= None, eos_token="<eos>", use_vocab= True, tokenize = 'spacy', batch_first= True, is_target=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
src_reader = reader('/content/drive/My Drive/kp20ktrain.src')
tgt_reader = reader('/content/drive/My Drive/kp20ktrain.tgt')
examples = []
fields = {'src': src_field, 'tgt': tgt_filed}
for s,t in zip(src_reader, tgt_reader):
   
    src=   json.loads(s)['src']
    tgts =  json.loads(t)['tgt']
    for tgt in tgts[:2]:
      ex = {}
      # print(tgt)
      ex['src'] = src
      ex['tgt'] = tgt  
      ex_fields = {k: [(k, v)] for k, v in fields.items() }
      examples.append(Example.fromdict(ex,ex_fields) )

dataset = torchtext.data.Dataset(examples, fields)
del examples
src_field.build_vocab(dataset.src, dataset.tgt)
tgt_filed.vocab = src_field.vocab
dataiter = torchtext.data.Iterator(dataset, batch_size=64, sort_key=lambda x: len(x.src), shuffle=True)


model = NMTModel(len(src_field.vocab), embedding_dim=100, hidden_dim=256, vocab=src_field.vocab, device = device, bidirectional=True)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
criterion = torch.nn.NLLLoss(ignore_index=src_field.vocab.stoi['<pad>'])
model.train()
epochs =10

for i in range(epochs):
  epoch_loss = 0
  for data in dataiter:
      inp = data.src.to(device)
      target = data.tgt.to(device)
      torch.cuda.empty_cache()
      outputs = model(inp, target)
      loss = criterion(outputs[0].view(-1, len(src_field.vocab)), target.view(-1))
      loss.backward()
      # torch.nn.utils.clip_grad_norm(model.parameters(), 2.0)
      optimizer.step()

      epoch_loss += loss.item()
  print(i, epoch_loss)
  


# In[ ]:


source_text = ["manual and gaze input cascaded ( magic ) pointing . this work explores a new direction in utilizing eye gaze for computer input . gaze tracking has long been considered as an alternative or potentially superior pointing method for computer input . we believe that many fundamental limitations exist with traditional gaze pointing . in particular , it is unnatural to overload a perceptual channel such as vision with a motor control task . we therefore propose an alternative approach , dubbed magic ( manual and gaze input cascaded ) pointing . with such an approach , pointing appears to the user to be a manual task , used for fine manipulation and selection . however , a large portion of the cursor movement is eliminated by warping the cursor to the eye gaze area , which encompasses the target . two specific magic pointing techniques , one conservative and one liberal , were designed , analyzed , and implemented with an eye tracker we developed . they were then tested in a pilot study . this early stage exploration showed that the magic pointing techniques might offer many advantages , including reduced physical effort and fatigue as compared to traditional manual pointing , greater accuracy and naturalness than traditional gaze pointing , and possibly faster speed than manual pointing . the pros and cons of the two techniques are discussed in light of both performance data and subjective reports"]
inp = src_field.tokenize(source_text[0])
inp = src_field.numericalize([inp]).to(device)


# In[ ]:


result = []
enc_output = model.encoder(inp)
  


# In[ ]:


res = model.decoder.infer_rnn_auto_regressive(encoder_output_dict=enc_output,vocab=src_field.vocab, length= 3).view(-1).detach().cpu().numpy()


# In[2]:


for i in res:
  print(src_field.vocab.itos[int(i)])


# In[ ]:


src_field.vocab.stoi['shot change detection']


# In[ ]:


import pickle
with open('vocab', 'wb') as fp:
  pickle.dump(src_field.vocab, fp)


# In[ ]:


torch.save(model.state_dict(), '/content/drive/My Drive/nmtmodel2.pt')


# In[ ]:


# import pickle
# with open('/content/vocab.pkl', 'wb') as fp:
#   pickle.dump(src_field.vocab, fp)


# In[3]:


#Inference loading and running model
# import nltk
# nltk.download('punkt')
import pickle
from nmtmodel import NMTModel
vocab = pickle.load(open('vocab', 'rb'))
model = NMTModel(len(vocab), embedding_dim=100, hidden_dim=256, vocab=vocab, device = torch.device('cpu'), bidirectional=True)
model.load_state_dict(torch.load('/content/drive/My Drive/nmtmodel2.pt', map_location= torch.device('cpu')))
model.eval()
from nltk.tokenize import word_tokenize


source_text = ["feasibility of a primarily digital research library . this position paper explores the issues related to the feasibility of having a primarily digital research library support the teaching and research needs of a university. the asian university for women (auw), a new university in chittagong, bangladesh, will open in september 2009. it must make a decision regarding the investment to be made in research resources to support the university. mass digitization efforts now make it possible to consider establishing a research library that consists primarily of digital resources rather than print. there are, however, many issues that make this consideration quite complex and far from certain. in this paper we explore the issues at a preliminary level. we focus on four broad perspectives in order to begin addressing the complex interactions that must be considered in transitioning to a primarily digital research environment: technical, economic, policy and social issues. the purpose of this paper is to begin to explore a research agenda for transitioning from a model for libraries where resources are primarily print to one that is predominantly digital. our research in this area is just beginning, so our purpose is to raise the issues rather than offer firm conclusions."]
inp = word_tokenize(source_text[0])
inp = [vocab.stoi[word] for word in inp]
result = []
enc_output = model.encoder(torch.tensor(inp).unsqueeze(0))

res = model.decoder.infer_rnn_auto_regressive(encoder_output_dict=enc_output,vocab=src_field.vocab, length= 3).view(-1).detach().cpu().numpy()
for i in res:
  print(vocab.itos[int(i)])

