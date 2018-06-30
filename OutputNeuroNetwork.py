# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 00:34:23 2018

@author: front
"""

def OutputNeuroNetwork(output):
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
        
        return answer