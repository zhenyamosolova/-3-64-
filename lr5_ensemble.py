"""
ЛАБОРАТОРНАЯ РАБОТА №5
Тема: Ансамблевые методы машинного обучения (бэггинг и бустинг)
Датасет: Wine (встроенный в scikit-learn)
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

# ======================== ЧАСТЬ 1. ЗАГРУЗКА И АНАЛИЗ ДАННЫХ ========================
print("="*70)
print("ЛР5. АНСАМБЛЕВЫЕ МЕТОДЫ: БЭГГИНГ И БУСТИНГ")
print("="*70)

print("\n▶ ЧАСТЬ 1. ЗАГРУЗКА И ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ")
print("-"*70)

# Загрузка данных
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print(f"\n✓ Датасет: Wine (вина)")
print(f"  - Источник: UCI Machine Learning Repository")
print(f"  - Количество образцов: {X.shape[0]}")
print(f"  - Количество признаков: {X.shape[1]}")
print(f"  - Количество классов: {y.nunique()} (class_0, class_1, class_2)")
print(f"  - Пропуски: {X.isnull().sum().sum()}")

print(f"\n✓ Список признаков:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i:2}. {col}")

print(f"\n✓ Распределение целевой переменной:")
for cls, count in y.value_counts().items():
    print(f"  - Класс {cls} ({data.target_names[cls]}): {count} образцов ({count/len(y)*100:.1f}%)")

# ======================== ЧАСТЬ 2. ПРЕДОБРАБОТКА ========================
print("\n▶ ЧАСТЬ 2. ПРЕДОБРАБОТКА ДАННЫХ")
print("-"*70)

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)
print("✓ Выполнена стандартизация признаков (StandardScaler)")

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% на тест
    random_state=42,    # для воспроизводимости
    stratify=y          # стратификация
)

print(f"\n✓ Разделение выборки:")
print(f"  - Обучающая выборка: {X_train.shape[0]} образцов ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  - Тестовая выборка:   {X_test.shape[0]} образцов ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"  - Стратификация: выполнена (пропорции классов сохранены)")

# ======================== ЧАСТЬ 3. СОЗДАНИЕ МОДЕЛЕЙ ========================
print("\n▶ ЧАСТЬ 3. СОЗДАНИЕ АНСАМБЛЕВЫХ МОДЕЛЕЙ")
print("-"*70)

# Базовая модель для бэггинга
base_tree = DecisionTreeClassifier(max_depth=5, random_state=42)

# Словарь с моделями (разбиваем на бэггинг и бустинг)
bagging_models = {
    '1. Bagging (бэггинг)': BaggingClassifier(
        estimator=base_tree, 
        n_estimators=50, 
        random_state=42
    ),
    '2. Random Forest (случайный лес)': RandomForestClassifier(
        n_estimators=50, 
        max_depth=5, 
        random_state=42
    ),
    '3. Extra Trees (сверхслучайные деревья)': ExtraTreesClassifier(
        n_estimators=50, 
        max_depth=5, 
        random_state=42
    )
}

boosting_models = {
    '4. AdaBoost (адаптивный бустинг)': AdaBoostClassifier(
        n_estimators=50, 
        random_state=42
    ),
    '5. Gradient Boosting (градиентный бустинг)': GradientBoostingClassifier(
        n_estimators=50, 
        max_depth=3, 
        random_state=42
    )
}

# Объединяем все модели
all_models = {**bagging_models, **boosting_models}

print(f"\n✓ Создано 5 моделей:")
print("\n  ГРУППА БЭГГИНГА (3 модели):")
for name in bagging_models.keys():
    print(f"    • {name}")
print("\n  ГРУППА БУСТИНГА (2 модели):")
for name in boosting_models.keys():
    print(f"    • {name}")

# ======================== ЧАСТЬ 4. ОБУЧЕНИЕ И ОЦЕНКА ========================
print("\n▶ ЧАСТЬ 4. ОБУЧЕНИЕ И ОЦЕНКА КАЧЕСТВА")
print("-"*70)

results = {}
training_times = {}

for name, model in all_models.items():
    print(f"\n{'─'*60}")
    print(f">>> МОДЕЛЬ: {name}")
    print('─'*60)
    
    # Замер времени обучения
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    training_times[name] = train_time
    
    # Предсказания
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Метрики качества
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    results[name] = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'time': train_time
    }
    
    print(f"\n  ⏱ Время обучения: {train_time:.3f} секунд")
    print(f"\n  📊 Точность (Accuracy):")
    print(f"     - На обучающей выборке: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"     - На тестовой выборке:   {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"     - Разница:               {train_acc - test_acc:.4f}")
    
    print(f"\n  📋 Отчет классификации (тестовая выборка):")
    print(classification_report(y_test, y_test_pred, target_names=data.target_names))
    
    print(f"\n  🔢 Матрица ошибок (тестовая выборка):")
    cm = confusion_matrix(y_test, y_test_pred)
    print("         Предсказано")
    print("         ", " ".join([f"{data.target_names[i][:10]:>10}" for i in range(3)]))
    for i in range(3):
        print(f"  Было {data.target_names[i][:10]:<10} {cm[i]}")

# ======================== ЧАСТЬ 5. СРАВНИТЕЛЬНЫЙ АНАЛИЗ ========================
print("\n" + "="*70)
print("ЧАСТЬ 5. СРАВНИТЕЛЬНЫЙ АНАЛИЗ МОДЕЛЕЙ")
print("="*70)

# Таблица результатов
print("\n▶ ТАБЛИЦА 1. Сравнение всех моделей")
print("-"*70)
print(f"\n{'Модель':<35} {'Точность(тест)':^15} {'Точность(обуч)':^15} {'Разница':^10} {'Время(c)':^10}")
print("-"*85)

# Сортируем по точности на тесте
sorted_models = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)

for i, (name, metrics) in enumerate(sorted_models):
    medal = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "  "))
    print(f"{medal} {name:<33} {metrics['test_accuracy']:<15.4f} {metrics['train_accuracy']:<15.4f} "
          f"{metrics['train_accuracy']-metrics['test_accuracy']:<10.4f} {metrics['time']:<10.3f}")

# Сравнение групп методов
print("\n▶ ТАБЛИЦА 2. Сравнение групп методов (бэггинг vs бустинг)")
print("-"*70)

bagging_scores = [results[name]['test_accuracy'] for name in bagging_models.keys()]
boosting_scores = [results[name]['test_accuracy'] for name in boosting_models.keys()]

print(f"\n  ГРУППА БЭГГИНГА (3 модели):")
print(f"    - Лучшая точность: {max(bagging_scores):.4f}")
print(f"    - Худшая точность: {min(bagging_scores):.4f}")
print(f"    - Средняя точность: {np.mean(bagging_scores):.4f}")
print(f"    - Стандартное отклонение: {np.std(bagging_scores):.4f}")

print(f"\n  ГРУППА БУСТИНГА (2 модели):")
print(f"    - Лучшая точность: {max(boosting_scores):.4f}")
print(f"    - Худшая точность: {min(boosting_scores):.4f}")
print(f"    - Средняя точность: {np.mean(boosting_scores):.4f}")
print(f"    - Стандартное отклонение: {np.std(boosting_scores):.4f}")

# Анализ переобучения
print("\n▶ ТАБЛИЦА 3. Анализ переобучения")
print("-"*70)

print("\n  Критерий: разница между точностью на обучении и тесте > 0.1 = переобучение")
print(f"\n  {'Модель':<35} {'Разница':^12} {'Вердикт':^15}")
print("-"*62)

for name, metrics in results.items():
    diff = metrics['train_accuracy'] - metrics['test_accuracy']
    if diff > 0.1:
        verdict = "⚠ ПЕРЕОБУЧЕНИЕ"
    elif diff > 0.05:
        verdict = "⚠ Небольшое переобучение"
    else:
        verdict = "✓ OK"
    print(f"  {name:<33} {diff:<12.4f} {verdict:^15}")

# ======================== ЧАСТЬ 6. ВЫВОДЫ ========================
print("\n" + "="*70)
print("ЧАСТЬ 6. ВЫВОДЫ ПО РЕЗУЛЬТАТАМ РАБОТЫ")
print("="*70)

# Лучшая модель
best_model_name = sorted_models[0][0]
best_model_score = sorted_models[0][1]['test_accuracy']

print(f"\n1. ЛУЧШАЯ МОДЕЛЬ:")
print(f"   → {best_model_name}")
print(f"   → Точность на тестовой выборке: {best_model_score:.4f} ({best_model_score*100:.2f}%)")

# Сравнение бэггинг vs бустинг
avg_bagging = np.mean(bagging_scores)
avg_boosting = np.mean(boosting_scores)

print(f"\n2. СРАВНЕНИЕ ГРУПП МЕТОДОВ:")
if avg_boosting > avg_bagging:
    print(f"   → Бустинг показал результат ЛУЧШЕ на {avg_boosting - avg_bagging:.4f}")
elif avg_boosting < avg_bagging:
    print(f"   → Бэггинг показал результат ЛУЧШЕ на {avg_bagging - avg_boosting:.4f}")
else:
    print(f"   → Результаты групп равны")

# Время обучения
print(f"\n3. СКОРОСТЬ ОБУЧЕНИЯ:")
fastest_model = min(training_times.items(), key=lambda x: x[1])
slowest_model = max(training_times.items(), key=lambda x: x[1])
print(f"   → Самая быстрая: {fastest_model[0]} ({fastest_model[1]:.3f} сек)")
print(f"   → Самая медленная: {slowest_model[0]} ({slowest_model[1]:.3f} сек)")

# Итоговая таблица
print("\n" + "="*70)
print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("="*70)

summary_df = pd.DataFrame([
    {
        'Модель': name,
        'Точность (тест)': metrics['test_accuracy'],
        'Точность (обуч)': metrics['train_accuracy'],
        'Разница': metrics['train_accuracy'] - metrics['test_accuracy'],
        'Время (с)': metrics['time']
    }
    for name, metrics in results.items()
]).sort_values('Точность (тест)', ascending=False)

print("\n")
print(summary_df.to_string(index=False))

print("\n" + "="*70)
print("Лабораторная работа №5 выполнена успешно!")
print("="*70)