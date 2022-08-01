import torch
from torch import nn
#from torch.utils.data import Dataset, DataLoader
#from torch.optim import Adam

import pandas as pd
import numpy as np
import os

from utils import MODEL_CLASSES, MODEL_PATH_MAP #SPECIAL_TOKENS

data_path=os.getcwd()+'/../../dataset/'
max_tokenizer_length = 100 #100 #20 
#WiC 데이터는 (SENTENCE1,SENTENCE2) 문장길이가 길어서 maxlength가 넘으면 잘라줬음. 
# 나중에 넘는건 제외하는 식으로 바꿔야함.

model_name = 'koelectra' #'kobert', 'roberta-base', 'koelectra', 'koelectra_QA', 'koelectra_tunib'
config_class, model_class, _ = MODEL_CLASSES[model_name] #config_class, model_class, model_tokenizer
model_path = MODEL_PATH_MAP[model_name]
modelClass_config = config_class.from_pretrained(model_path)

## COLA model ##
# 문장 문법성 판단
# input : 768차원의 임베딩 벡터
# OutPut은 2개 (1은 맞음, 0은 틀림)

class model_COLA(nn.Module):
    def __init__(self):
        super(model_COLA, self).__init__()
        self.model_PLM = model_class.from_pretrained(model_path) #[10,100]*3 -> [10,100,768]    #PLM (Pre-trained Language Model)
        self.relu = nn.ReLU()   # activation function
        self.linear = nn.Linear(768,2) #CLS: 200->2(binary)

    def forward(self, input_ids, token_type_ids, attention_mask):
    # MyModel에 input(input_ids,token_type_ids,attention_mask) 텐서들이 들어왔을때 forward를 실행.
        iIds = input_ids.long() #.unsqueeze(0).long()        # input ids
        tok_typeIds = token_type_ids.long() #.unsqueeze(0).long()       # token type
        attMask = attention_mask.long() #.unsqueeze(0)      # attention mask
        
        output=self.model_PLM(input_ids=iIds, token_type_ids=tok_typeIds, attention_mask=attMask)
        #torch.Size([batch_size, max_len, 768])


        output=output['last_hidden_state'][:, 0, :]     # batch size 전부, max 중 첫번째 (CLS Token), attention mask 전부 슬라이싱
        #CLS token: max_len의 길이 토큰 중 첫번째(0번째) 토큰의 마지막 레이어만 임베딩으로 사용

        self.relu(output)       # activation function
        output=self.linear(output) #768->2
        
        return output

## Wic model ##
class model_WiC_biSent(nn.Module):
    def __init__(self):
        super(model_WiC_biSent, self).__init__()
        self.model_PLM = model_class.from_pretrained(model_path)
        
        self.linear = nn.Linear(768,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, token_type_ids, attention_mask): #MyModel에 input(input_ids,token_type_ids,attention_mask) 텐서들이 들어왔을때 forward를 실행.
        iIds1 = input_ids.long()
        tok_typeIds1 = token_type_ids.long()
        attMask1 = attention_mask.long()

        output=self.model_PLM(input_ids=iIds1, token_type_ids=tok_typeIds1, attention_mask=attMask1) 
        
        output = output['last_hidden_state'][:, 0, :]

        self.sigmoid(output) #output의 tensor 소숫값을 확률값(0~1)로 바꿔줌. (bs,768) -> (bs,768)
        output = self.linear(output) #tensor: 768 -> 1 #linear classifier

        #output = self.softmax(output)
        #print(output.shape)
        
        return output
    
## COPA model ##
class model_COPA_biSent(nn.Module):
    def __init__(self):
        super(model_COPA_biSent, self).__init__()
        self.model_PLM = model_class.from_pretrained(model_path)

        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(768,1) #CLS: 768->1 (model-output concat해서 실질적으론 2개)

    def forward(self, input_ids, token_type_ids, attention_mask): #MyModel에 input(input_ids,token_type_ids,attention_mask) 텐서들이 들어왔을때 forward를 실행.
        
        iIds = input_ids.long() #.unsqueeze(0).long()
        tok_typeIds = token_type_ids.long() #.unsqueeze(0).long()
        attMask = attention_mask.long() #.unsqueeze(0)
        
        output=self.model_PLM(input_ids=iIds, token_type_ids=tok_typeIds, attention_mask=attMask) #token_type_ids=tok_typeIds, 
        #print('output(bert-lhs)[CLS]: ',output['last_hidden_state'][:, 0, :].shape) #CLS token: torch.Size([batch_size, 768])

        #CLS토큰임베딩&LinearFC 사용하는 경우
        output = output['last_hidden_state'][:, 0, :] #cls_embedding: (bs,768)
        self.sigmoid(output) #output의 tensor 소숫값을 확률값(0~1)로 바꿔줌. (bs,768) -> (bs,768)
        output = self.linear(output) #tensor: 768 -> 1 #linear classifier

        return output

## BoolQA model ##
class model_BoolQA(nn.Module):
    def __init__(self):
        super(model_BoolQA, self).__init__()
        self.model_PLM = model_class.from_pretrained(model_path, config=modelClass_config) #[10,100]*3 -> [10,100,768]
        self.relu = nn.ReLU() # Activation Func.
        self.linear = nn.Linear(768,2) #CLS: 200->2(binary)

    def forward(self, input_ids, token_type_ids, attention_mask): #MyModel에 input(input_ids,token_type_ids,attention_mask) 텐서들이 들어왔을때 forward를 실행.
        iIds = input_ids.long()#.unsqueeze(0).long()
        tok_typeIds = token_type_ids.long()#.unsqueeze(0).long()
        attMask = attention_mask.long()#.unsqueeze(0)

        output=self.model_PLM(input_ids=iIds, token_type_ids=tok_typeIds, attention_mask=attMask) #token_type_ids=tok_typeIds,

        output=output['last_hidden_state'][:, 0, :]

        self.relu(output)
        output=self.linear(output) #768->2
        
        return output
    
## TODO: WiC model (abandoned) ##
class model_WiC_biSent_abandoned(nn.Module):
    def __init__(self):
        super(model_WiC_biSent_abandoned, self).__init__()
        #self.bert1 = BertModel.from_pretrained('monologg/kobert') #PreTrainedModel(input_ids, token_type_ids, attention_mask): [10,100]*3 -> [10,100,768]
        #self.bert = BertModel.from_pretrained('monologg/kobert')
        self.model_PLM = model_class.from_pretrained(model_path)
        
        #분류에 token embedding을 활용하는 경우
        #self.pooling = nn.AvgPool1d(3)
        
        #self.relu = nn.ReLU() #활성화 함수(0~1): nn.sigmoid, nn.relu(-를 0으로)
        #self.sigmoid = nn.Sigmoid()
        self.cosSim = nn.CosineSimilarity(dim=1, eps=1e-6)
        
        #output1,2 합쳐서 결과 도출 (768,1)
        self.linear = nn.Linear(768,1) #CLS: 768->2(binary)
        #self.pooling = nn.Linear(768,1)
        #self.softmax = nn.Softmax(dim=1) #dim=0(세로합=1),dim=1(가로합=1) #nn.LogSoftmax()

        #word_index는 찾으려는 단어의 위치 
    def forward(self, input_ids1, token_type_ids1, attention_mask1, word_index1,input_ids2, token_type_ids2, attention_mask2, word_index2): #MyModel에 input(input_ids,token_type_ids,attention_mask) 텐서들이 들어왔을때 forward를 실행.
        iIds1 = input_ids1.long()#.unsqueeze(0).long()
        tok_typeIds1 = token_type_ids1.long()#.unsqueeze(0).long()
        attMask1 = attention_mask1.long()#.unsqueeze(0)

        iIds2 = input_ids2.long()#.unsqueeze(0).long()
        tok_typeIds2 = token_type_ids2.long()#.unsqueeze(0).long()
        attMask2 = attention_mask2.long()#.unsqueeze(0)
 
        #output1과 output2의 결과를 따로 계산
        output1=self.model_PLM(input_ids1=iIds1, token_type_ids1=tok_typeIds1, attention_mask1=attMask1)
        output2=self.model_PLM(input_ids2=iIds2, token_type_ids1=tok_typeIds2, attention_mask2=attMask2)

        #print('output1 :',output1,'output2: ', output2)

        #기존 주석 
        #print('output(bert-lhs)[CLS]: ',output['last_hidden_state'][:, 0, :].shape) #CLS token: torch.Size([batch_size, 768])
        #print('output shape: ',output['last_hidden_state'].shape, tokIdx_start.shape, tokIdx_end.shape) 
        # #torch.Size([bs, 100, 768]), torch.Size([100])*2개

        output1=output1['last_hidden_state'][:,word_index1,:] #해당 단어만 사용 
        output2=output2['last_hidden_state'][:,word_index2,:]

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        #output1과 output2 코사인유사도 비교 
        output = cos(output1, output2)

        #output=self.linear(768,1)
        #print('output : ',output)
        #print(output.shape)

        #코사인 유사도 비교후 -> 바로 그 값 리턴 
        return output
    
    