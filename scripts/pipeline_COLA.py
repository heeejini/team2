import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import pandas as pd
import numpy as np
import os

from pprint import pprint
from datetime import datetime

from transformers import AdamW

from utils import compute_metrics,  MCC, get_label, set_seed
from utils import MODEL_PATH_MAP
from utils import TOKEN_MAX_LENGTH #SPECIAL_TOKENS
from utils import getParentPath, save_model, load_model, save_json, DATASET_PATHS #print_timeNow


from kobert_datasets import COLA_dataset
from kobert_models import model_COLA

data_path=os.getcwd()+'/../../dataset/'
model_name = 'koelectra' #'kobert', 'roberta-base', 'koelectra'
task_name = 'COLA' #'COLA', 'WiC', 'COPA', 'BoolQ'

taskDir_path, fname_train, fname_dev, fname_test, _ = DATASET_PATHS[task_name]

data_path=os.getcwd()+'/../../dataset/' # 24번줄이랑 중복
max_tokenizer_length = TOKEN_MAX_LENGTH[task_name] #100 #20 #WiC 데이터는 (SENTENCE1,SENTENCE2) 문장길이가 길어서 maxlength가 넘으면 잘라줬음. 나중에 넘는건 제외하는 식으로 바꿔야함.

countEpoch = 0

bestMCC = -2 #-1 ~ +1       # 매튜 상관 계수 (Matthews Correlation Coefficient, MCC) : 1에 가까울수록 비슷 (옳고 그름 판별 이진분류 metric)
bestAcc = -1 #0 ~ 1
bestLoss = 1 #0 ~ 1
bestLoss_at = 0

def train_cola(model, data_loader, batch_size, epochs, lf, optimizer, device):
    model.train() #set model training mode: gradient 업데이트(O)

    min_loss = 1 #initial value(0~1)

    for _ in range(epochs):     # 정해진 epochs 수만큼 반복 train
        correct = 0
        all_loss = []
        mini_batch = 0
        print(f'[epoch {countEpoch+_}]') #print(f'[epoch {_}]')     # 몇번째 epoch인지 카운트해서 print

        for batchIdx, (input_ids, token_type_ids, attention_mask, label) in enumerate(data_loader):     # label까지 해서 훈련
            #print(input_ids.shape, token_type_ids.shape, attention_mask.shape, label)
            #batch inputs shape: #max_len=64(TOKEN_MAX_LENGTH), bs=20(pipeline_main) 으로 세팅
            #   - torch.Size([max_len, bs])x3개 (input_ids, token_type_ids, attention_mask)
            #   - tensor([0, 1, 0, 0, 1, 1, 0, 1, 1, 0]) (label) ->len=max_len

            model.zero_grad() #model-params optimizer의 gradient를 0으로 초기화.
            # 항상 backpropagation을 하기전에 gradients를 zero로 만들어주고 시작을 해야된다 (미분값이 누적되지 않도록)

            #device = torch.device('cuda:0') #device: 'cpu' 'cuda:0' 'cuda:1'
            #model.to(device)

            #move param_buffers from cpu to gpu     (.to(device) == .cuda())
            input_ids = input_ids.to(device) #torch.Size([10, bs])
            token_type_ids = token_type_ids.to(device) #torch.Size([10, bs])
            attention_mask = attention_mask.to(device) #torch.Size([10, bs])
            label = label.long().to(device) #tensor([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])     # 정답 데이터 (맞는지 틀리는지)

            output = model(input_ids, token_type_ids, attention_mask) #torch.Size([10, bs]) -> torch.Size([10, 2])
            # 모델 돌림

            lf_input=output #torch.Size([10, 2])
            lf_target=label #torch.Size([10])
            loss = lf(output,label) #torch.Size([20, 2]), tensor        # lf는 loss function. 학습한 output과 정답 라벨 사이의 loss를 구해줌

            print(output[0])
            print(label)
            exit()
            
            pred = torch.argmax(output,dim=-1) #torch.Size([10,1]) #model-output 중 큰 값의 인덱스 선택(0/1)
            # output은 2차원(0일 확률, 1일 확률), 이 중 확률이 큰거 선택
            correct += sum(pred.detach().cpu()==label.detach().cpu())       # 정답을 맞췄으면 correct += 1, 틀렸으면 그대로.
            all_loss.append(loss)       # 모든 loss를 이어 붙임
            loss.backward() #자동으로 모든 기울기 계산
            optimizer.step() #가중치 업데이트
            mini_batch += batch_size
            #print(mini_batch,"/",len(colaDataset)) #batch학습 진행률: batch/전체Dataset 
        
        #print(sum(all_loss)/len(all_loss))
        #print("acc = ", correct / len(colaDataset))
        avg_loss = (sum(all_loss)/len(all_loss)).detach().cpu().float()
        accuracy = (correct / len(colaDataset_train)).float()
        print("acc = ", accuracy,", loss = ",avg_loss)
        
        #bestLoss = min(bestLoss, avg_loss)
        min_loss = min(min_loss, avg_loss)

    return min_loss
#   return min_loss
def eval_cola(model, data_loader, batch_size, device):      # Dev set(Validation set)
    model.eval() #set model eval mode: gradient 업데이트(X)

    y_true = None #label list
    y_pred = None #model prediction list

    for batchIdx, (input_ids, token_type_ids, attention_mask, label) in enumerate(data_loader):     # data_loader는 batch size에 맞게 반복문 돌릴 수 있도록 감싼 형태
        with torch.no_grad(): #autograd 끔->속도향상. (Option: model.eval()하면 안해줘도 됨.) # 라고 적혀있는데 해줘야된다고 함.
                                                # model.eval()은 Dropout, Batchnorm등의 기능 비활성화 -> 추론모드 (메모리랑 관련 X)
                                                # torch.no_grad()는 autograd engine을 비활성화 -> 필요한 메모리 줄여주고 연산속도 증가 (dropout을 비활성화시키진 않음)
            #move param_buffers to gpu
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.long().to(device)
            # 4개의 파라미터들 다 gpu 1번카드로 옮겨줌.

            output = model(input_ids, token_type_ids, attention_mask)   # WordPiece Embedding, Segment Embedding, Attention Mask
            
            logits = output #torch.argmax(output,dim=-1)

        if y_pred is None:
            y_pred = logits.detach().cpu().numpy()
            y_true = label.detach().cpu().numpy()
            # detach()는 연산 기록으로 부터 분리한 tensor을 반환하는 method
            # cpu()는 GPU 메모리에 올려져 있는 tensor를 cpu 메모리로 복사하는 method
            # GPU 메모리에 올려져 있는 tensor를 numpy로 변환하기 위해

        else:
            y_pred = np.append(y_pred, logits.detach().cpu().numpy(), axis=0)
            y_true = np.append(y_true, label.detach().cpu().numpy(), axis=0)

    y_pred = np.argmax(y_pred, axis=1)      # 최대값에 해당하는 index 찾기
    result = MCC(y_pred, y_true)        # 매튜 상관 계수 (Matthews Correlation Coefficient, MCC) 계산 : 1에 가까울수록 비슷
    accuracy = compute_metrics(y_pred, y_true)["acc"]       # 정확도 측정
    #print('eval_MCC = ',result)
    #print('eval_acc = ',accuracy)
    print('eval_MCC = ',result,', eval_acc = ',accuracy)

    return result, accuracy
#   return result, accuracy

def inference_cola(model, data_loader, batch_size, device):
    model.eval()

    y_pred = None #model prediction list

    for batchIdx, (input_ids, token_type_ids, attention_mask, label) in enumerate(data_loader):
        with torch.no_grad(): #autograd 끔->속도향상. 사실 model.eval()하면 안해줘도 됨.
            input_ids = input_ids.to(device) #move param_buffers to gpu
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)

            output = model(input_ids, token_type_ids, attention_mask)
            
            logits = output#torch.argmax(output,dim=-1)

        if y_pred is None:
            y_pred = logits.detach().cpu().numpy()
        else:
            y_pred = np.append(y_pred, logits.detach().cpu().numpy(), axis=0)
    y_pred = np.argmax(y_pred, axis=1)

    print('output shape: ',y_pred.shape, type(y_pred))

    return y_pred
#   return y_pred

##monologg/KoBERT##
if __name__ == "__main__":
    homePth = getParentPath(os.getcwd())
    datasetPth = homePth+'/dataset/'
    print('homePth:',homePth,', curPth:',os.getcwd())
    #start_day_time=print_timeNow()
    #print('training start at (date, time): ',print_timeNow())

    tsvPth_train = datasetPth+taskDir_path+fname_train  #'task1_grammar/NIKL_CoLA_train.tsv'
    tsvPth_dev = datasetPth+taskDir_path+fname_dev #'task1_grammar/NIKL_CoLA_dev.tsv'
    tsvPth_test = datasetPth+taskDir_path+fname_test #'task1_grammar/NIKL_CoLA_test.tsv' 

    bs = 100 #gpu 가능한 bs: [10,20,100,200,400]    #batch size : 400
    epochs= 20 #50#2#100 #시도한 epochs: [10, 100]
    num_printEval = 4 #4 #꼭 epochs의 약수가 되게 넣어주기. 안그럼 epochs가 조금 모자라게 돔.

    device = torch.device('cuda:1')     # 1번 그래픽카드 사용
    model_type = 'uniBert' # 'uniBert', 'biBERT'
    random_seed_int = 5 # 랜덤시드 넘버=5 로 고정
    set_seed(random_seed_int, device) #random seed 정수로 고정.     # seed를 항상 고정해놓아야 함. 안그러면 매번 결과가 달라짐. 모델의 학습 결과를 Reproduction 하기 위해

    bool_save_model, bool_load_model, bool_save_output = False, False, False #default: True, False, False
    ## model save/load path & save_output_path ##

    lf = nn.CrossEntropyLoss()
    # Cross Entropy Loss는 보통 Classification에서 많이 사용됨.
    # 최종값이 나오고 Softmax 함수를 통해 이 값들의 범위는 [0,1], 총 합은 1이 되도록
    # 그 다음, 1-hot Label (정답 라벨)과의 Cross Entropy를 통해 Loss를 구함

    mymodel = model_COLA()      # scripts\kobert_models.py의 model_COLA()
    mymodel.to(device)      # 1번 그래픽카드로 모델을 불러옴

    optimizer = Adam(mymodel.parameters(),lr=2e-5, eps=1e-8)    # Optimizer는 딥러닝에서 Network가 빠르고 정확하게 학습하는 것을 목표로 함
    #optimizer = AdamW(mymodel.parameters(), lr=1e-5)

    colaDataset_train = COLA_dataset(os.path.join(os.getcwd(),tsvPth_train)) #(dataPth_train)
    colaDataset_dev = COLA_dataset(os.path.join(os.getcwd(),tsvPth_dev)) #(dataPth_dev)
    colaDataset_test = COLA_dataset(os.path.join(os.getcwd(),tsvPth_test)) #(dataPth_test)
    
    # DataLoader : Dataset 을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 감싼 것.
    TrainLoader = DataLoader(colaDataset_train, batch_size=bs)
    EvalLoader = DataLoader(colaDataset_dev, batch_size=bs)
    InferenceLoader = DataLoader(colaDataset_test, batch_size=bs)

    print('[Training Phase]')
    print(f'len {task_name}_train:{len(colaDataset_train)}, batch_size:{bs}, epochs:{epochs}(eval_by {num_printEval}), device({device})')
    for epoch in range(int(epochs/num_printEval)):
        # 평가
        result, accuracy = eval_cola(mymodel, EvalLoader, bs, device)       # mcc(옳은지 : 1 ~ 틀린지 : 0), 정확도
        print(f'before epoch{epoch}: devSet(MCC:{result:.4f}, acc:{accuracy:.4f})') #
        #mccTest, accTest = eval_cola(mymodel, InferenceLoader, bs, device) #
        #print(f'before epoch{epoch}: testSet(MCC:{mccTest:.4f}, acc:{accTest:.4f})') #
        bestMCC = max(bestMCC,result)
        bestAcc = max(bestAcc,accuracy)
        # mcc나 정확도가 커지면 업데이트

        minLoss = train_cola(mymodel, TrainLoader, bs, num_printEval, lf, optimizer, device) #4epoch마다 eval
        if minLoss < bestLoss:
            bestLoss = minLoss
            bestLoss_at = countEpoch
        countEpoch+=num_printEval #countEpoch 업데이트

    print('[Evaluation Phase]')
    print(f'len {task_name}_dev:{len(colaDataset_dev)}, batch_size:{bs}, epochs:{epochs}, device({device})')
    result, accuracy = eval_cola(mymodel, EvalLoader, bs, device)
    bestMCC = max(bestMCC,result)
    bestAcc = max(bestAcc,accuracy)
    #print(f'Dev - bestMCC:{bestMCC}, bestAccuracy:{bestAcc}, bestLoss:{bestLoss}')

    print('[Inference Phase]')
    eval_cola(mymodel, InferenceLoader, bs, device) #test acc 결과뽑기.
    modelOutput = inference_cola(mymodel, InferenceLoader, bs, device)

    ## TODO: save model path ##

    #end_day_time=print_timeNow()
    #print(f'training model from {start_day_time} to {end_day_time} (date, time): ')
    print('finish')
    print('<SUMMARY>')
    print(f'task:{task_name}, model:{model_name}({model_type}), bs:{bs}, epochs:{epochs}, load/save model:{bool_load_model}/{bool_save_model}, randSeedNum:{random_seed_int}')
    print(f'bestAccuracy:{bestAcc}, bestMCC:{bestMCC}, bestLoss:{bestLoss}(bestLoss around epoch {bestLoss_at})')

    print('end main')
