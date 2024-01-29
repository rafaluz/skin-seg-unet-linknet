# Automatic Segmentation of Melanoma Skin Cancer Using Transfer Learning and Fine-tuning
By  [Rafael](https://github.com/rafaluz), [Flávio](https://github.com/flaviohdaraujo) and [Romuere](https://github.com/romuere/).

[Access the full article](https://link.springer.com/article/10.1007/s00530-021-00840-3)

## Abstract:
The massive use of multimedia technologies has enabled the exploration of information in many data such as texts, audio, videos, and images. Computational methods are being developed for several purposes such as monitoring, security, business, and even health through the automatic diagnosis of diseases by medical images. Among these diseases, we have melanoma skin cancer. Melanoma is a skin cancer that causes a large number of fatalities worldwide. Several methods for the automatic diagnosis of melanoma in dermoscopic images have been developed. For these methods to be more efficient, it is essential to isolate the lesion region. This study used a melanoma segmentation method based on U-net and Linknet deep learning networks combined with transfer learning and fine-tuning techniques. Additionally, we evaluate the model's ability to learn to segment the disease or just the dataset by combining datasets. The experiments were carried out in three datasets (PH2, ISIC 2018, and DermIS) and obtained promising results, with emphasis on the U-net that obtained an average Dice of 0.923 in the PH2 dataset, Dice = 0.893 in ISIC 2018, and Dice = 0.879 in the DermIS dataset.

## Method:
![MULTIMEDIASYSTEMSMETODOLOGIA](https://user-images.githubusercontent.com/12652832/118539510-e0e97000-b725-11eb-8ac6-20b60bf0bead.png)

## Tests:
<img src="https://user-images.githubusercontent.com/12652832/118540099-9fa59000-b726-11eb-9656-18fe403d052e.png" width="600px">

## Results:
![image](https://user-images.githubusercontent.com/12652832/118539999-84d31b80-b726-11eb-9448-a5e1371ca8a0.png)

## Instructions for using the algorithm
Create a virtual environment and install the dependencies     
> pip install -r requeriments.txt

Open the * .py file and set values for:

- Choose the GPU at:
    > GPU_GLOBAL
- Choose the Dataset (0 - ph2, 1 - dermis, 2 - isic2018) in:
    > base_escolhida
- Choose the folder where the tests will be saved (don't forget to create the folder before running the code):
    > PASTA_DE_TESTES
- Choose batch_size at:
    > BATCH_SIZE_GLOBAL
- Choose the number of seasons in:
    > NUMERO_EPOCAS_GLOBAL 

Run the *.py file:
> python *.py
