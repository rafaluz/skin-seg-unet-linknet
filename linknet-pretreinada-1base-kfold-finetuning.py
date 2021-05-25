
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
import time
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import os
from segmentation_models.utils import set_trainable
from keras.optimizers import Adam

########################## PARAMETROS ##########################
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#QUAL GPU VOCÊ QUER USAR?
GPU_GLOBAL = 0

#QUAL A BASE? (0-ph2, 1-dermis, 2-isic2018)
base_escolhida = 0

# PASTA DOS TESTES (NÃO ESQUEÇA DE CRIAR)
# teste1-ph2 teste2-dermis teste3-isic2018
PASTA_DE_TESTES = 'TESTES/LINKNET/FINETUNING/teste1-ph2/'
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


 ################## CHAMAR BASE ESCOLHIDA
imagens, mascaras_medico = escolhe_base(base_escolhida)


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



########################################################

inicio = time.time()

num_folds = 5

sensitivity_results_test = []
specificity_results_test = []
accuracy_results_test = []
auc_results_test = []
dice_results_test = []
jaccard_results_test = []

RESULTADOS = []

sensitivity_results_test_finetuning = []
specificity_results_test_finetuning = []
accuracy_results_test_finetuning = []
auc_results_test_finetuning = []
dice_results_test_finetuning = []
jaccard_results_test_finetuning = []

RESULTADOS_finetuning = []


# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(imagens, mascaras_medico):
    print("######### KFOLD ",fold_no,"#########")
    
    
    ###### CRIA O MODELO
    BACKBONE = 'resnet34'
    preprocess_input = sm.get_preprocessing(BACKBONE)
    # define model
    model = sm.Linknet(BACKBONE, encoder_weights='imagenet', encoder_freeze=True)
    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )


  
    ###### dividir treino e validação, lembrando que como são 5 folds, tem 80% pra trein e 20% para test. Então dos 80% de treino pego 25% para validação. E ai mantenho a mesma proporção de antes do kfold (60% treino, 20% teste e 20% validação)
    x_train, x_val, y_train, y_val = train_test_split(imagens[train], mascaras_medico[train], test_size = 0.25, random_state = 11)

    x_train= np.asarray(x_train)
    y_train= (np.asarray(y_train)> threshold_otsu(np.asarray(y_train)))
    x_val= np.asarray(x_val)
    y_val= (np.asarray(y_val)>threshold_otsu(np.asarray(y_val)))
    x_test= np.asarray(imagens[test])
    y_test= (np.asarray(mascaras_medico[test])>threshold_otsu(np.asarray(mascaras_medico[test])))
    
    y_train = y_train.reshape(-1, 128, 128, 1) #.astype('float32') 
    y_val = y_val.reshape(-1, 128, 128, 1) #.astype('float32') 
    y_test = y_test.reshape(-1, 128, 128, 1) #.astype('float32') 
    


    callbacks = [
        keras.callbacks.ModelCheckpoint(str(PASTA_DE_TESTES)+'best_model'+str(fold_no)+'.h5', save_weights_only=True, save_best_only=True, mode='min'),
#         keras.callbacks.ReduceLROnPlateau(),
    ]

    ###### TREINA ######
    model.fit(
       x=x_train,
       y=y_train,
       batch_size=BATCH_SIZE_GLOBAL,
       epochs=NUMERO_EPOCAS_GLOBAL,
       callbacks=callbacks, 
       validation_data=(x_val, y_val),
    )

    ###### TESTA normal
    predicoes = model.predict(x_test)
    
    predicoes= (np.asarray(predicoes)> threshold_otsu(np.asarray(predicoes)))
    
    # calcular metricas do teste
    sensitivity,specificity,accuracy,auc,dice,jaccard = calc_metric(y_test,predicoes[:,:,:,0])
    print("dice ", dice)
    
    sensitivity_results_test.append(sensitivity)
    specificity_results_test.append(specificity)
    accuracy_results_test.append(accuracy)
    auc_results_test.append(auc)
    dice_results_test.append(dice)
    jaccard_results_test.append(jaccard)
    
    print("---- TESTE NORMAL - FOLD ",fold_no)
    print("sensitivity:",sensitivity)
    print("specificity:",specificity)
    print("accuracy:",accuracy)
    print("auc:",auc)
    print("dice:",dice)
    print("jaccard:",jaccard)


    ################################### Ajuste fino ###################################
    model.optimizer=Adam(lr=0.00001)
    # release all layers for training
    set_trainable(model) # set all layers trainable and recompile model


    callbacks = [
        keras.callbacks.ModelCheckpoint(str(PASTA_DE_TESTES)+'best_model_finetuning'+str(fold_no)+'.h5', save_weights_only=True, save_best_only=True, mode='min'),
#         keras.callbacks.ReduceLROnPlateau(),
    ]
    ###### AJUSTE FINO ###### # continue training
    model.fit(
       x=x_train,
       y=y_train,
       batch_size=BATCH_SIZE_GLOBAL,
       epochs=75,
       callbacks=callbacks, 
       validation_data=(x_val, y_val),
    )
    
    ###### TESTA FINETUNING
    predicoes = model.predict(x_test)
    
    predicoes= (np.asarray(predicoes)> threshold_otsu(np.asarray(predicoes)))
    
    # calcular metricas do teste
    sensitivity,specificity,accuracy,auc,dice,jaccard = calc_metric(y_test,predicoes[:,:,:,0])
    print("dice ", dice)
    
    sensitivity_results_test_finetuning.append(sensitivity)
    specificity_results_test_finetuning.append(specificity)
    accuracy_results_test_finetuning.append(accuracy)
    auc_results_test_finetuning.append(auc)
    dice_results_test_finetuning.append(dice)
    jaccard_results_test_finetuning.append(jaccard)
    
    print("---- TESTE - FOLD ",fold_no)
    print("sensitivity:",sensitivity)
    print("specificity:",specificity)
    print("accuracy:",accuracy)
    print("auc:",auc)
    print("dice:",dice)
    print("jaccard:",jaccard)
    
    # Increase fold number
    fold_no = fold_no + 1
    
    keras.backend.clear_session()

print("############## RESULTADO FINAL ##############")
print("---- MEDIAS DO TESTE ------")
print("sensitivity:",np.mean(sensitivity_results_test))
print("specificity:",np.mean(specificity_results_test))
print("accuracy:",np.mean(accuracy_results_test))
print("auc:",np.mean(auc_results_test))
print("dice:",np.mean(dice_results_test))
print("jaccard:",np.mean(jaccard_results_test))
print(PASTA_DE_TESTES)

np.savetxt(str(PASTA_DE_TESTES)+"sensitivity_results_test.csv", sensitivity_results_test, delimiter=",")
np.savetxt(str(PASTA_DE_TESTES)+"specificity_results_test.csv", specificity_results_test, delimiter=",")
np.savetxt(str(PASTA_DE_TESTES)+"accuracy_results_test.csv", accuracy_results_test, delimiter=",")
np.savetxt(str(PASTA_DE_TESTES)+"auc_results_test.csv", auc_results_test, delimiter=",")
np.savetxt(str(PASTA_DE_TESTES)+"dice_results_test.csv", dice_results_test, delimiter=",")
np.savetxt(str(PASTA_DE_TESTES)+"jaccard_results_test.csv", jaccard_results_test, delimiter=",")
print("CSVs salvos")

fim = time.time()
tempo_processamento = fim-inicio
print("TEMPO DE PROCESSAMENTO: ",tempo_processamento)


# SALVAR RESULTADOS GERAIS
teste_results = pd.Series([np.mean(sensitivity_results_test), np.mean(specificity_results_test), np.mean(accuracy_results_test), np.mean(auc_results_test), np.mean(dice_results_test), np.mean(jaccard_results_test),tempo_processamento])
resultados = pd.DataFrame([list(teste_results)],  columns =  ["Sensitivity", "specificity", "accuracy", "auc", "dice", "jaccard","tempo"])
np.savetxt(str(PASTA_DE_TESTES)+"RESULTADOS.csv",resultados,fmt='%.16f', delimiter=",")

# SALVAR DESVIO PADRAO 
teste_results_desvio_padrao = pd.Series([np.std(sensitivity_results_test), np.std(specificity_results_test), np.std(accuracy_results_test), np.std(auc_results_test), np.std(dice_results_test), np.std(jaccard_results_test)])
resultados_desvio_padrao = pd.DataFrame([list(teste_results_desvio_padrao)],  columns =  ["Sensitivity", "specificity", "accuracy", "auc", "dice", "jaccard"])
np.savetxt(str(PASTA_DE_TESTES)+"RESULTADOS-DESVIO-PADRAO.csv",resultados_desvio_padrao,fmt='%.16f', delimiter=",")


# ################# FINETUNING
print("############## RESULTADO FINAL FINETUNING ##############")
print("---- MEDIAS DO TESTE FINETUNING ------")
print("sensitivity:",np.mean(sensitivity_results_test_finetuning))
print("specificity:",np.mean(specificity_results_test_finetuning))
print("accuracy:",np.mean(accuracy_results_test_finetuning))
print("auc:",np.mean(auc_results_test_finetuning))
print("dice:",np.mean(dice_results_test_finetuning))
print("jaccard:",np.mean(jaccard_results_test_finetuning))
print(PASTA_DE_TESTES)


np.savetxt(str(PASTA_DE_TESTES)+"sensitivity_results_test_finetuning.csv", sensitivity_results_test_finetuning, delimiter=",")
np.savetxt(str(PASTA_DE_TESTES)+"specificity_results_test_finetuning.csv", specificity_results_test_finetuning, delimiter=",")
np.savetxt(str(PASTA_DE_TESTES)+"accuracy_results_test_finetuning.csv", accuracy_results_test_finetuning, delimiter=",")
np.savetxt(str(PASTA_DE_TESTES)+"auc_results_test_finetuning.csv", auc_results_test_finetuning, delimiter=",")
np.savetxt(str(PASTA_DE_TESTES)+"dice_results_test_finetuning.csv", dice_results_test_finetuning, delimiter=",")
np.savetxt(str(PASTA_DE_TESTES)+"jaccard_results_test_finetuning.csv", jaccard_results_test_finetuning, delimiter=",")
print("CSVs salvos")

# SALVAR RESULTADOS GERAIS
teste_results_finetuning = pd.Series([np.mean(sensitivity_results_test_finetuning), np.mean(specificity_results_test_finetuning), np.mean(accuracy_results_test_finetuning), np.mean(auc_results_test_finetuning), np.mean(dice_results_test_finetuning), np.mean(jaccard_results_test_finetuning),tempo_processamento])
resultados_finetuning = pd.DataFrame([list(teste_results_finetuning)],  columns =  ["Sensitivity", "specificity", "accuracy", "auc", "dice", "jaccard","tempo"])
np.savetxt(str(PASTA_DE_TESTES)+"RESULTADOS_finetuning.csv",resultados_finetuning,fmt='%.16f', delimiter=",")

# SALVAR DESVIO PADRAO 
teste_results_desvio_padrao = pd.Series([np.std(sensitivity_results_test_finetuning), np.std(specificity_results_test_finetuning), np.std(accuracy_results_test_finetuning), np.std(auc_results_test_finetuning), np.std(dice_results_test_finetuning), np.std(jaccard_results_test_finetuning)])
resultados_desvio_padrao = pd.DataFrame([list(teste_results_desvio_padrao)],  columns =  ["Sensitivity", "specificity", "accuracy", "auc", "dice", "jaccard"])
np.savetxt(str(PASTA_DE_TESTES)+"RESULTADOS_finetuning-DESVIO-PADRAO.csv",resultados_desvio_padrao,fmt='%.16f', delimiter=",")