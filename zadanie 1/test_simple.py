#!/usr/bin/env python3
"""
Простой тестовый скрипт для проверки работы
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic():
    print(" Тестирование базовых функций...")
    
    # Тест 1: Проверка импортов
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        print(" Библиотеки импортированы")
    except ImportError as e:
        print(f" Ошибка импорта: {e}")
        return False
    
    # Тест 2: Создание тестовых данных
    try:
        data = {
            'event_type': ['view', 'cart', 'purchase'],
            'price': [10.0, 20.0, 30.0],
            'user_id': ['user1', 'user2', 'user3']
        }
        df = pd.DataFrame(data)
        print(f" Тестовые данные созданы: {df.shape}")
        return True
    except Exception as e:
        print(f" Ошибка создания данных: {e}")
        return False

if __name__ == "__main__":
    if test_basic():
        print("\n Все базовые тесты пройдены!")
        sys.exit(0)
    else:
        print("\n Тесты не пройдены")
        sys.exit(1)