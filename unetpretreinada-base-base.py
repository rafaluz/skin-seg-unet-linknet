import tensorflow as tf
import keras

import segmentation_models as sm

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.io import imread, imread_collection, imsave
#from scipy.misc import imsave as save
from skimage.filters import median,threshold_otsu
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import time

########################## PARAMETROS ##########################

#QUAL GPU VOCÊ QUER USAR?
GPU_GLOBAL = 0

#QUAL A BASE? (0-ph2, 1-dermis, 2-isic2018)
base_escolhida1 = 0
base_escolhida2 = 1

# PASTA DOS TESTES 
# teste4-ph2-dermis teste5-ph2-isic18 
# teste6-isic18-ph2 teste7-isic18-dermis
# teste8-dermis-ph2 teste9-dermis-isic18
PASTA_DE_TESTES = 'TESTES/UNET/teste4-ph2-dermis/'
print("##########################")
print(PASTA_DE_TESTES)


if not os.path.exists(PASTA_DE_TESTES):
    os.makedirs(PASTA_DE_TESTES)

BATCH_SIZE_GLOBAL = 48
NUMERO_EPOCAS_GLOBAL = 150
###############################################################

def escolhe_base(base_escolhida):
    if base_escolhida == 0:
        # ph2
        imagens = imread_collection('IMAGENS/PH2PROPORCIONAL128/imagens/*')
        mascaras_medico = imread_collection('IMAGENS/PH2PROPORCIONAL128/mascaras/*')
    elif base_escolhida == 1:
        melanoma_imagens = imread_collection('IMAGENS/DERMIS128/melanoma/*orig*')
        melanoma_mascaras_medico = imread_collection('IMAGENS/DERMIS128/melanoma/*contour*')

        notmelanoma_imagens = imread_collection('IMAGENS/DERMIS128/notmelanoma/*orig*')
        notmelanoma_mascaras_medico = imread_collection('IMAGENS/DERMIS128/notmelanoma/*contour*')

        imagens = np.concatenate((melanoma_imagens, notmelanoma_imagens), axis=0)
        mascaras_medico = np.concatenate((melanoma_mascaras_medico, notmelanoma_mascaras_medico), axis=0)
    elif base_escolhida == 2:
        melanoma_imagens = imread_collection('IMAGENS/ISIC2018-128/MELANOMA/*')
        melanoma_mascaras_medico = imread_collection('IMAGENS/ISIC2018-128/MASKMELANOMA/*')

        notmelanoma_imagens = imread_collection('IMAGENS/ISIC2018-128/NMELANOMA/*')
        notmelanoma_mascaras_medico = imread_collection('IMAGENS/ISIC2018-128/MASKNMELANOMA/*')

        imagens = np.concatenate((melanoma_imagens, notmelanoma_imagens), axis=0)
        mascaras_medico = np.concatenate((melanoma_mascaras_medico, notmelanoma_mascaras_medico), axis=0)
    else:
        print(" Escolha uma base de imagens")
        
    
    return np.array(imagens), np.array(mascaras_medico)



# 1 = base de treino e 2 = base de teste
def treinoEteste(imagens1,mascaras1,imagens2,mascaras2):
    
    # test
    x_test = np.asarray(imagens2)
    y_test = (np.asarray(mascaras2)> threshold_otsu(np.asarray(mascaras2)))
    
    
    # train e validate
    x_train, x_val, y_train, y_val = train_test_split(imagens1, mascaras1, test_size = 0.2, random_state = 12)
    
    x_val= np.asarray(x_val)
    y_val= (np.asarray(y_val)>threshold_otsu(np.asarray(y_val)))
    x_train= np.asarray(x_train)
    y_train= (np.asarray(y_train)>threshold_otsu(np.asarray(y_train)))
    
    y_train = y_train.reshape(-1, 128, 128, 1) #.astype('float32') 
    y_val = y_val.reshape(-1, 128, 128, 1) #.astype('float32') 
    y_test = y_test.reshape(-1, 128, 128, 1) #.astype('float32') 
    
    return x_train, y_train, x_val, y_val, x_test, y_test


# ESCOLHER BASE 1
imagens1, mascaras1 = escolhe_base(base_escolhida1)
# ESCOLHER BASE 2
imagens2, mascaras2 = escolhe_base(base_escolhida2)


# treino 1 - teste 2
x_train, y_train, x_val, y_val, x_test, y_test = treinoEteste(imagens1,mascaras1,imagens2,mascaras2)



def calc_metric(y_true,y_pred):
    
    #padronizando o y_test
    y_true = np.expand_dims(y_true,axis=-1)
    y_true = np.int64(y_true)
    
#     print(y_pred.shape,y_true.shape,np.unique(y_pred),np.unique(y_true))
#     print(y_pred)
    cm = confusion_matrix(y_true.ravel(),y_pred.ravel())
    tn, fp, fn, tp = cm.ravel()
    return calc_metrics_matrix(tn, fp, fn, tp)

def calc_metrics_matrix(tn, fp, fn, tp):
    dice = (2.0 * tp) / ((2.0 * tp) + fp + fn)
    jaccard = (1.0 * tp) / (tp + fp + fn) 
    sensitivity = (1.0 * tp) / (tp + fn)
    specificity = (1.0 * tn) / (tn + fp)
    accuracy = (1.0 * (tn + tp)) / (tn + fp + tp + fn)
    auc = 1 - 0.5 * (((1.0 * fp) / (fp + tn)) + ((1.0 * fn) / (fn + tp)))
    # prec = float(tp)/float(tp + fp)
    # fscore = float(2*tp)/float(2*tp + fp + fn)

    return sensitivity,specificity,accuracy,auc,dice,jaccard



# #################### RODAR COM A GPU ##################### (comentar tudo caso der erro)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[GPU_GLOBAL], 'GPU')
    except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)
print(gpus)




##################################################


BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet')
model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)


inicio = time.time()

####### treinamento
model.fit(
   x=x_train,
   y=y_train,
   batch_size=BATCH_SIZE_GLOBAL,
   epochs=NUMERO_EPOCAS_GLOBAL,
   validation_data=(x_val, y_val),
)


fim = time.time()
tempo_processamento = fim-inicio
print("TEMPO DE PROCESSAMENTO: ",tempo_processamento)



###### TESTA
predicoes = model.predict(x_test)

predicoes= (np.asarray(predicoes)> threshold_otsu(np.asarray(predicoes)))

# calcular metricas do teste
sensitivity,specificity,accuracy,auc,dice,jaccard = calc_metric(y_test,predicoes[:,:,:,0])


print("############## RESULTADO FINAL ##############")
print("sensitivity:",sensitivity)
print("specificity:",specificity)
print("accuracy:",accuracy)
print("auc:",auc)
print("dice:",dice)
print("jaccard:",jaccard)
print(PASTA_DE_TESTES)

# SALVAR RESULTADOS
teste_results = pd.Series([sensitivity, specificity, accuracy, auc, dice, jaccard,tempo_processamento])
resultados = pd.DataFrame([list(teste_results)],  columns =  ["Sensitivity", "specificity", "accuracy", "auc", "dice", "jaccard","tempo"])
np.savetxt(str(PASTA_DE_TESTES)+"RESULTADOS.csv",resultados,fmt='%.16f', delimiter=",")

