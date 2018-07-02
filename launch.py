# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 22:23:53 2018

@author: front
"""
import numpy as np
from keras.models import model_from_json
import test
import OutputNeuroNetwork as onn

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

answer = onn.OutputNeuroNetwork(output)
        
userAns = int(input('Даже если ответ нейронной сети совпадает,' 
              '\nвсё-равно введите ответ, '
              '\nэто поможет нейросети развиваться: '))

#Преобразуем ответ к нужному виду
Check = np.uint8([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
Check[0][userAns] = 1

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
        
        output = loaded_model.predict(X, batch_size=200, verbose=2, steps=None)
        
        answer = onn.OutputNeuroNetwork(output)
        
print('Нейросеть окончила свою работу')
        
        
    