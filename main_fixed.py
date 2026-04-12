"""
ЛАБОРАТОРНАЯ РАБОТА: Обработка данных
Выполнено с использованием ТРЁХ разных датасетов
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler

print("="*60)
print("ЛАБОРАТОРНАЯ РАБОТА: ПРЕДОБРАБОТКА ДАННЫХ")
print("="*60)

# ============================================================================
# ЧАСТЬ 1: ОБРАБОТКА ПРОПУСКОВ (Датасет: Пациенты больницы)
# ============================================================================

print("\n" + "█"*60)
print("ЧАСТЬ 1: ОБРАБОТКА ПРОПУСКОВ")
print("Датасет: 'Пациенты больницы'")
print("█"*60)

# Создаём датасет с пропусками разных типов
hospital_data = {
    'PatientID': [1, 2, 3, 4, 5],
    'Name': ['Иванов', 'Петров', None, 'Сидоров', 'Кузнецова'],  # Текстовый пропуск
    'Age': [45, 28, None, 52, 34],                                 # Числовой пропуск
    'BloodPressure': [120, None, 135, None, 118],                  # Числовой пропуск
    'Diagnosis': ['Грипп', 'Ангина', 'Грипп', None, 'Бронхит'],    # Категориальный пропуск
    'RoomType': ['Обычная', 'VIP', 'Обычная', 'Обычная', None]      # Ещё один категориальный
}

df_hospital = pd.DataFrame(hospital_data)
print("\nИсходные данные (с пропусками):")
print(df_hospital)
print("\nКоличество пропусков по колонкам:")
print(df_hospital.isnull().sum())

# --- Обработка пропусков разными стратегиями ---
print("\n" + "-"*40)
print("Обработка пропусков:")

# Для числовых колонок: используем МЕДИАНУ (устойчива к выбросам)
numeric_cols_hosp = ['Age', 'BloodPressure']
num_imputer = SimpleImputer(strategy='median')
df_hospital[numeric_cols_hosp] = num_imputer.fit_transform(df_hospital[numeric_cols_hosp])
print("✓ Числовые пропуски (Age, BP) → заполнены МЕДИАНОЙ")

# Для категориальных колонок: используем НАИБОЛЕЕ ЧАСТОЕ ЗНАЧЕНИЕ (моду)
categorical_cols_hosp = ['Name', 'Diagnosis', 'RoomType']
cat_imputer = SimpleImputer(strategy='most_frequent')
df_hospital[categorical_cols_hosp] = cat_imputer.fit_transform(df_hospital[categorical_cols_hosp])
print("✓ Категориальные пропуски → заполнены МОДОЙ")

print("\nРезультат после обработки пропусков:")
print(df_hospital)

# ============================================================================
# ЧАСТЬ 2: КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ (Датасет: Автомобили)
# ============================================================================

print("\n" + "█"*60)
print("ЧАСТЬ 2: КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
print("Датасет: 'Автомобили'")
print("█"*60)

# Создаём датасет с разными типами категориальных признаков
cars_data = {
    'CarID': [1, 2, 3, 4, 5],
    'Brand': ['Toyota', 'BMW', 'Toyota', 'Mercedes', 'BMW'],        # Номинативная (нет порядка)
    'Color': ['Red', 'Blue', 'Red', 'Black', 'White'],               # Номинативная
    'FuelType': ['Petrol', 'Diesel', 'Petrol', 'Electric', 'Diesel'], # Номинативная
    'Size': ['Small', 'Medium', 'Small', 'Large', 'Medium']          # ОРДИНАЛЬНАЯ (есть порядок)
}

df_cars = pd.DataFrame(cars_data)
print("\nИсходные данные (категориальные признаки):")
print(df_cars)

# --- 2A: One-Hot Encoding для номинативных признаков (нет порядка) ---
print("\n" + "-"*40)
print("2A. One-Hot Encoding (для бренда, цвета, типа топлива):")

nominal_cols = ['Brand', 'Color', 'FuelType']
encoder_onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_onehot = encoder_onehot.fit_transform(df_cars[nominal_cols])

# Создаём красивые названия колонок
onehot_columns = []
for col, values in zip(nominal_cols, encoder_onehot.categories_):
    for val in values:
        onehot_columns.append(f"{col}_{val}")

df_onehot = pd.DataFrame(encoded_onehot, columns=onehot_columns)
print("One-Hot результат:")
print(df_onehot)

# --- 2B: Ordinal Encoding для порядковых признаков (размер S < M < L) ---
print("\n" + "-"*40)
print("2B. Ordinal Encoding (для размера, где Small < Medium < Large):")

size_order = [['Small', 'Medium', 'Large']]  # Задаём порядок!
encoder_ordinal = OrdinalEncoder(categories=size_order)
df_cars['Size_Encoded'] = encoder_ordinal.fit_transform(df_cars[['Size']])

print("Результат Ordinal Encoding:")
print(df_cars[['CarID', 'Size', 'Size_Encoded']])
print("(Small→0, Medium→1, Large→2)")

# --- Собираем итоговый датасет ---
df_cars_final = pd.concat([df_cars[['CarID']], df_onehot, df_cars[['Size_Encoded']]], axis=1)
print("\nИТОГО: Полностью закодированный датасет об автомобилях:")
print(df_cars_final)

# ============================================================================
# ЧАСТЬ 3: МАСШТАБИРОВАНИЕ ДАННЫХ (Датасет: Недвижимость)
# ============================================================================

print("\n" + "█"*60)
print("ЧАСТЬ 3: МАСШТАБИРОВАНИЕ ДАННЫХ")
print("Датасет: 'Недвижимость'")
print("█"*60)

# Создаём датасет с признаками в РАЗНЫХ ЕДИНИЦАХ ИЗМЕРЕНИЯ
houses_data = {
    'HouseID': [1, 2, 3, 4, 5],
    'Area_m2': [45, 120, 68, 250, 89],        # от 45 до 250 (разброс небольшой)
    'Price_usd': [80000, 350000, 120000, 750000, 150000],  # от 80к до 750к (ОГРОМНЫЙ разброс!)
    'Rooms': [1, 3, 2, 5, 2],                 # от 1 до 5
    'YearBuilt': [2005, 2018, 2010, 2022, 2015]  # от 2005 до 2022
}

df_houses = pd.DataFrame(houses_data)
print("\nИсходные данные (разные масштабы):")
print(df_houses)
print("\nСтатистика исходных данных:")
print(df_houses[['Area_m2', 'Price_usd', 'Rooms', 'YearBuilt']].describe())

# --- 3A: StandardScaler (стандартизация) ---
print("\n" + "-"*40)
print("3A. StandardScaler (среднее=0, дисперсия=1):")

scaler_standard = StandardScaler()
numeric_features = ['Area_m2', 'Price_usd', 'Rooms', 'YearBuilt']
df_standard = df_houses.copy()
df_standard[numeric_features] = scaler_standard.fit_transform(df_houses[numeric_features])

print("После StandardScaler:")
print(df_standard)
print("\nСредние значения (должны быть ~0):")
print(df_standard[numeric_features].mean().round(2))
print("Стандартные отклонения (должны быть ~1):")
print(df_standard[numeric_features].std().round(2))

# --- 3B: MinMaxScaler (нормализация в диапазон [0, 1]) ---
print("\n" + "-"*40)
print("3B. MinMaxScaler (все значения в [0, 1]):")

scaler_minmax = MinMaxScaler()
df_minmax = df_houses.copy()
df_minmax[numeric_features] = scaler_minmax.fit_transform(df_houses[numeric_features])

print("После MinMaxScaler:")
print(df_minmax)
print("\nДиапазоны значений (мин=0, макс=1):")
for col in numeric_features:
    print(f"{col}: от {df_minmax[col].min():.2f} до {df_minmax[col].max():.2f}")

# ============================================================================
# ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ
# ============================================================================

print("\n" + "█"*60)
print("ИТОГОВЫЙ ВЫВОД")
print("█"*60)

summary = pd.DataFrame({
    'Задача': ['Обработка пропусков', 'Кодирование категорий', 'Масштабирование'],
    'Датасет': ['Пациенты больницы', 'Автомобили', 'Недвижимость'],
    'Основные методы': ['SimpleImputer (median, most_frequent)', 
                        'OneHotEncoder + OrdinalEncoder', 
                        'StandardScaler + MinMaxScaler'],
    'Ключевая идея': ['Замена NaN на осмысленные значения', 
                      'Превращение текста в числа', 
                      'Приведение к одному масштабу']
})
print(summary.to_string(index=False))

print("\n" + "="*60)
print("ЛАБОРАТОРНАЯ РАБОТА ВЫПОЛНЕНА УСПЕШНО!")
print("="*60)