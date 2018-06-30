# -*- coding: utf-8 -*-
"""
Created on Sun May 27 19:35:43 2018

@author: front
"""

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import model_from_json
import PhotoToBinMethod

print("Загружаю сеть из файлов")

# Загружаем данные об архитектуре сети
json_file = open("mnist_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

# Создаем модель
loaded_model = model_from_json(loaded_model_json)

# Загружаем сохраненные веса в модель
loaded_model.load_weights("mnist_model.h5")

print("Загрузка сети завершена")

# Загружаем данные
(x, y) = PhotoToBinMethod.PhotoBinaryAndAnswer()

x = x.reshape(60000, 784)

