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
from keras.datasets import mnist

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
        
userAns = int(input('Даже если ответ нейронной сети совпадает,' 
              '\nвсё-равно введите ответ, '
              '\nэто поможет нейросети развиваться: '))

//С этим костылём в дальнейшем разобраться...
if userAns == 0:
        Check = np.uint8([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
elif userAns == 1:
        Check = np.uint8([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
elif userAns == 2:
        Check = np.uint8([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
elif userAns == 3:
        Check = np.uint8([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
elif userAns == 4:
        Check = np.uint8([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
elif userAns == 5:
        Check = np.uint8([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
elif userAns == 6:
        Check = np.uint8([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
elif userAns == 7:
        Check = np.uint8([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])
elif userAns == 8:
        Check = np.uint8([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
elif userAns == 9:
        Check = np.uint8([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

if userAns != answer:
        print('Нейросеть обучается...')

        loaded_model.fit(X, Check, epochs=2, batch_size=200)
        
        print("Сохраняем сеть")

        # Сохраняем сеть для последующего использования
        # Генерируем описание модели в формате json
        model_json = loaded_model.to_json()
        json_file = open("mnist_model.json", "w")
        
        # Записываем архитектуру сети в файл
        json_file.write(model_json)
        json_file.close()
        
        # Записываем данные о весах в файл
        loaded_model.save_weights("mnist_model.h5")
        
        print("Сохранение сети завершено")
    