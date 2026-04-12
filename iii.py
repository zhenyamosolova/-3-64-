import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("=== Информация о датасете ===")
print(f"Размер: {df.shape}")
print(f"\nПервые 5 строк:")
print(df.head())
print(f"\nСтатистика:")
print(df.describe())
print(f"\nПроверка на пропуски:")
print(df.isnull().sum())
X = df[iris.feature_names].copy()
y = df['target'].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nМасштабирование выполнено")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.3, 
    random_state=42,
    stratify=y
)

print(f"\nОбучающая выборка: {X_train.shape[0]} образцов")
print(f"Тестовая выборка: {X_test.shape[0]} образцов")
k_initial = 5
knn_initial = KNeighborsClassifier(n_neighbors=k_initial)
knn_initial.fit(X_train, y_train)
y_pred_initial = knn_initial.predict(X_test)

print(f"\n=== БАЗОВАЯ МОДЕЛЬ (K={k_initial}) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_initial):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_initial, average='macro'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_initial, average='macro'):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_initial, average='macro'):.4f}")
param_grid = {'n_neighbors': range(1, 31, 2)}
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=kfold,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\n=== GRID SEARCH ===")
print(f"Лучший K: {grid_search.best_params_['n_neighbors']}")
print(f"Лучшая точность: {grid_search.best_score_:.4f}")

# График
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(10, 5))
plt.plot(results['param_n_neighbors'], results['mean_test_score'], 'bo-')
plt.xlabel('K (количество соседей)')
plt.ylabel('Accuracy')
plt.title('GridSearchCV: Зависимость качества от K')
plt.grid(True)
plt.show()
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_dist = {
    'n_neighbors': range(1, 51),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

random_search = RandomizedSearchCV(
    KNeighborsClassifier(),
    param_dist,
    n_iter=30,
    cv=stratified_kfold,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"\n=== RANDOMIZED SEARCH ===")
print(f"Лучшие параметры: {random_search.best_params_}")
print(f"Лучшая точность: {random_search.best_score_:.4f}")
best_knn = grid_search.best_estimator_
y_pred_best = best_knn.predict(X_test)

print(f"\n=== ОПТИМАЛЬНАЯ МОДЕЛЬ (K={grid_search.best_params_['n_neighbors']}) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_best, average='macro'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_best, average='macro'):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred_best, average='macro'):.4f}")

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title('Матрица ошибок')
plt.show()
print("\n" + "="*50)
print("СРАВНЕНИЕ МЕТРИК")
print("="*50)

print(f"{'Метрика':<20} {'Базовая (K=5)':<20} {'Оптимальная':<20} {'Изменение':<15}")
print("-"*75)
print(f"{'Accuracy':<20} {accuracy_score(y_test, y_pred_initial):<20.4f} {accuracy_score(y_test, y_pred_best):<20.4f} {(accuracy_score(y_test, y_pred_best)-accuracy_score(y_test, y_pred_initial)):+.4f}")
print(f"{'Precision':<20} {precision_score(y_test, y_pred_initial, average='macro'):<20.4f} {precision_score(y_test, y_pred_best, average='macro'):<20.4f} {(precision_score(y_test, y_pred_best, average='macro')-precision_score(y_test, y_pred_initial, average='macro')):+.4f}")
print(f"{'Recall':<20} {recall_score(y_test, y_pred_initial, average='macro'):<20.4f} {recall_score(y_test, y_pred_best, average='macro'):<20.4f} {(recall_score(y_test, y_pred_best, average='macro')-recall_score(y_test, y_pred_initial, average='macro')):+.4f}")
print(f"{'F1-Score':<20} {f1_score(y_test, y_pred_initial, average='macro'):<20.4f} {f1_score(y_test, y_pred_best, average='macro'):<20.4f} {(f1_score(y_test, y_pred_best, average='macro')-f1_score(y_test, y_pred_initial, average='macro')):+.4f}")
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
base_scores = [
    accuracy_score(y_test, y_pred_initial),
    precision_score(y_test, y_pred_initial, average='macro'),
    recall_score(y_test, y_pred_initial, average='macro'),
    f1_score(y_test, y_pred_initial, average='macro')
]
best_scores = [
    accuracy_score(y_test, y_pred_best),
    precision_score(y_test, y_pred_best, average='macro'),
    recall_score(y_test, y_pred_best, average='macro'),
    f1_score(y_test, y_pred_best, average='macro')
]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, base_scores, width, label=f'Базовая (K={k_initial})', color='skyblue')
bars2 = ax.bar(x + width/2, best_scores, width, label=f'Оптимальная (K={grid_search.best_params_["n_neighbors"]})', color='lightcoral')

ax.set_ylabel('Score')
ax.set_title('Сравнение качества моделей')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.set_ylim(0, 1.1)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

plt.tight_layout()
plt.show()
k_values = range(1, 31)
cv_scores_kfold = []
cv_scores_stratified = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores_kfold = cross_val_score(knn, X_train, y_train, cv=KFold(5, shuffle=True, random_state=42), scoring='accuracy')
    scores_stratified = cross_val_score(knn, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='accuracy')
    cv_scores_kfold.append(scores_kfold.mean())
    cv_scores_stratified.append(scores_stratified.mean())

plt.figure(figsize=(12, 6))
plt.plot(k_values, cv_scores_kfold, 'o-', label='K-Fold CV', linewidth=2)
plt.plot(k_values, cv_scores_stratified, 's-', label='Stratified K-Fold CV', linewidth=2)
plt.axvline(x=grid_search.best_params_['n_neighbors'], color='green', linestyle='--', label=f"Оптимальный K={grid_search.best_params_['n_neighbors']}")
plt.xlabel('K')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Сравнение стратегий кросс-валидации')
plt.legend()
plt.grid(True)
plt.show()
print("\n" + "="*60)
print("ИТОГИ ЛАБОРАТОРНОЙ РАБОТЫ")
print("="*60)
print(f"Датасет: Iris (150 образцов, 3 класса)")
print(f"Разделение: train/test = 70/30")
print(f"Масштабирование: StandardScaler")
print(f"\nБазовая модель: K={k_initial}")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_initial):.4f}")
print(f"\nОптимальная модель: K={grid_search.best_params_['n_neighbors']}")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"\nУлучшение: +{(accuracy_score(y_test, y_pred_best)-accuracy_score(y_test, y_pred_initial))*100:.2f}%")
print(f"\nИспользованные стратегии CV:")
print(f"  1. GridSearchCV + K-Fold (5 фолдов)")
print(f"  2. RandomizedSearchCV + Stratified K-Fold (5 фолдов)")
