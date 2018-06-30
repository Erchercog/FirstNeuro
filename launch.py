# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 22:23:53 2018

@author: front
"""
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import model_from_json
import test

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
X = test.main()

# Компилируем загруженную модель
loaded_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

# Оцениваем качество обучения сети загруженной сети на тестовых данных
output = loaded_model.predict(X, batch_size=200, verbose=2, steps=None)

#output = output / 255

kat = 0 
i = 0
while i < 10:
        if output[0,i] > kat:
                kat = output[0,i]
                answer = i
        print(i, " = ", round(output[0,i] * 100, 2), '%')
        i += 1

if output[0, answer] * 100 > 75:
        print('')
        print("Нейросеть уверена, что это - ", answer)
elif output[0, answer] * 100 > 65:
        print('')
        print("Нейросеть считает, что это - ", answer)
else:
        print('')
        print("Нейросеть предполагает, что это - ", answer)