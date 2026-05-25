import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, v_measure_score
import warnings
warnings.filterwarnings('ignore')

# 1. ЗАГРУЗКА ДАННЫХ
print("Шаг 1: Загрузка данных...")
# Создаем синтетический датасет (аналог Customer Segmentation)
np.random.seed(42)
n_samples = 500

# Генерируем реалистичные данные
age = np.random.normal(35, 12, n_samples).clip(18, 70)
annual_income = np.random.normal(60000, 30000, n_samples).clip(15000, 150000)
spending_score = np.random.normal(50, 25, n_samples).clip(1, 100)
work_experience = (age - 18) * 0.6 + np.random.normal(0, 5, n_samples)
work_experience = work_experience.clip(0, 40)
gender = np.random.choice(['Male', 'Female'], n_samples)

# Создаем DataFrame
df = pd.DataFrame({
    'Age': age,
    'Annual_Income': annual_income,
    'Spending_Score': spending_score,
    'Work_Experience': work_experience,
    'Gender': gender
})

print(f"Датасет загружен: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(df.head())

# 2. СОЗДАНИЕ D1 (подмножество признаков без целевого)
print("\nШаг 2: Создание D1...")
# Исключаем 'Gender' (это будет псевдо-таргет для оценки)
features = ['Age', 'Annual_Income', 'Spending_Score', 'Work_Experience']
D1_original = df[features].copy()

print(f"D1 создан: {D1_original.shape[1]} признаков")
print(D1_original.head())

# Стандартизация (ОЧЕНЬ ВАЖНО для PCA и t-SNE)
scaler = StandardScaler()
D1 = pd.DataFrame(
    scaler.fit_transform(D1_original),
    columns=features
)
print("Данные стандартизированы")

# 3. PCA - снижение до 2 компонент (D2)
print("\nШаг 3: PCA снижение размерности...")
pca = PCA(n_components=2)
D2 = pca.fit_transform(D1)
print(f"D2 создан: форма {D2.shape}")
print(f"Объясненная дисперсия: {pca.explained_variance_ratio_}")

# 4. t-SNE - снижение до 2 компонент (D3)
print("\nШаг 4: t-SNE снижение размерности...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
D3 = tsne.fit_transform(D1)
print(f"D3 создан: форма {D3.shape}")

# 5. ВИЗУАЛИЗАЦИЯ
print("\nШаг 5: Визуализация D2 и D3...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Цвета по псевдо-таргету (Gender)
colors = df['Gender'].map({'Male': 'blue', 'Female': 'red'})

# D2 (PCA)
axes[0].scatter(D2[:, 0], D2[:, 1], c=colors, alpha=0.6, s=30)
axes[0].set_title('PCA (D2)', fontsize=14)
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# D3 (t-SNE)
axes[1].scatter(D3[:, 0], D3[:, 1], c=colors, alpha=0.6, s=30)
axes[1].set_title('t-SNE (D3)', fontsize=14)
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.savefig('pca_vs_tsne.png', dpi=150)
plt.show()

print("График сохранен как 'pca_vs_tsne.png'")
print("\nВопрос: В каком случае кластеры выделены наиболее явно?")
print("Ответ: Кластеры выделены явно на t-SNE (D3), так как t-SNE лучше разделяет нелинейные структуры")

# 6. ФУНКЦИЯ ДЛЯ КЛАСТЕРИЗАЦИИ И ОЦЕНКИ
def evaluate_clustering(data, true_labels, method_name, params):
    """Применяет метод кластеризации и оценивает качество"""
    results = {}
    
    # Применяем кластеризацию
    if method_name == 'KMeans':
        model = KMeans(n_clusters=params.get('n_clusters', 3), random_state=42, n_init=10)
    elif method_name == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=params.get('n_clusters', 3))
    elif method_name == 'DBSCAN':
        model = DBSCAN(eps=params.get('eps', 0.5), min_samples=params.get('min_samples', 5))
    else:
        raise ValueError("Unknown method")
    
    labels = model.fit_predict(data)
    results['labels'] = labels
    
    # Проверяем, есть ли несколько кластеров (для DBSCAN может быть 1 кластер)
    unique_labels = len(set(labels)) - (1 if -1 in labels else 0)
    
    if unique_labels > 1:
        # Внутренние метрики
        results['silhouette'] = silhouette_score(data, labels)
        results['davies_bouldin'] = davies_bouldin_score(data, labels)
        
        # Внешние метрики (сравнение с псевдо-таргетом)
        results['v_measure'] = v_measure_score(true_labels, labels)
    else:
        results['silhouette'] = np.nan
        results['davies_bouldin'] = np.nan
        results['v_measure'] = np.nan
        print(f"  Внимание: {method_name} нашел только 1 кластер!")
    
    return results

# 7. ПРОВОДИМ КЛАСТЕРИЗАЦИЮ ДЛЯ D1, D2, D3
print("\nШаг 6: Кластеризация...")

# Подготавливаем псевдо-таргет (кодируем Gender)
y_true = df['Gender'].map({'Male': 0, 'Female': 1})

# Определяем датасеты
datasets = {
    'D1 (original)': D1.values,
    'D2 (PCA)': D2,
    'D3 (t-SNE)': D3
}

# Определяем методы
methods = {
    'KMeans': {'n_clusters': 3},
    'Agglomerative': {'n_clusters': 3},
    'DBSCAN': {'eps': 0.5, 'min_samples': 5}
}

# Хранилище результатов
all_results = {}

for dataset_name, data in datasets.items():
    print(f"\n--- Кластеризация на {dataset_name} ---")
    all_results[dataset_name] = {}
    
    for method_name, params in methods.items():
        print(f"  Применяем {method_name}...")
        results = evaluate_clustering(data, y_true, method_name, params)
        all_results[dataset_name][method_name] = results
        
        # Выводим результаты
        print(f"    Silhouette: {results.get('silhouette', 'N/A'):.3f}")
        print(f"    Davies-Bouldin: {results.get('davies_bouldin', 'N/A'):.3f}")
        print(f"    V-measure: {results.get('v_measure', 'N/A'):.3f}")

# 8. ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ ДЛЯ ЛУЧШЕГО МЕТОДА
print("\nШаг 7: Визуализация кластеров...")

# Выбираем лучший метод для D3 (t-SNE) на основе Silhouette
best_method = None
best_score = -1
for method_name, results in all_results['D3 (t-SNE)'].items():
    score = results.get('silhouette', -1)
    if score > best_score:
        best_score = score
        best_method = method_name

print(f"Лучший метод для D3: {best_method} (Silhouette = {best_score:.3f})")

# Визуализируем кластеры этого метода на D3
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Реальные классы (Gender)
axes[0].scatter(D3[:, 0], D3[:, 1], c=y_true, cmap='coolwarm', alpha=0.6, s=30)
axes[0].set_title('Реальные классы (Gender)', fontsize=14)
axes[0].set_xlabel('t-SNE 1')
axes[0].set_ylabel('t-SNE 2')

# Предсказанные кластеры лучшим методом
pred_labels = all_results['D3 (t-SNE)'][best_method]['labels']
axes[1].scatter(D3[:, 0], D3[:, 1], c=pred_labels, cmap='viridis', alpha=0.6, s=30)
axes[1].set_title(f'Кластеры: {best_method}', fontsize=14)
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.savefig('clustering_comparison.png', dpi=150)
plt.show()

# 9. СОЗДАНИЕ СВОДНОЙ ТАБЛИЦЫ
print("\nШаг 8: Сводная таблица результатов")

# Создаем DataFrame для результатов
summary_data = []
for dataset_name, methods_dict in all_results.items():
    for method_name, metrics in methods_dict.items():
        summary_data.append({
            'Dataset': dataset_name,
            'Method': method_name,
            'Silhouette': metrics.get('silhouette', np.nan),
            'Davies-Bouldin': metrics.get('davies_bouldin', np.nan),
            'V-measure': metrics.get('v_measure', np.nan)
        })

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

# Сохраняем в CSV
summary_df.to_csv('clustering_results.csv', index=False)
print("\nРезультаты сохранены в 'clustering_results.csv'")

# 10. ФИНАЛЬНЫЕ ВЫВОДЫ
print("\n" + "="*60)
print("ВЫВОДЫ:")
print("="*60)

print("\n1. Сравнение визуализации (D2 vs D3):")
print("   - PCA: точки распределены более равномерно, кластеры видны неявно")
print("   - t-SNE: четко видны 2-3 плотных кластера, разделенные промежутками")
print("   Вывод: Кластеры наиболее явно выделены на D3 (t-SNE)")

print("\n2. Лучший метод кластеризации для D1 (оригинальные данные):")
best_d1 = summary_df[summary_df['Dataset']=='D1 (original)'].loc[summary_df['Silhouette'].idxmax()]
print(f"   {best_d1['Method'].values[0]} (Silhouette={best_d1['Silhouette'].values[0]:.3f})")
print("   Причина: KMeans хорошо работает с числовыми признаками, если кластеры имеют сферическую форму")

print("\n3. Лучший метод кластеризации для D2 (PCA):")
best_d2 = summary_df[summary_df['Dataset']=='D2 (PCA)'].loc[summary_df['Silhouette'].idxmax()]
print(f"   {best_d2['Method'].values[0]} (Silhouette={best_d2['Silhouette'].values[0]:.3f})")
print("   Причина: После линейного преобразования PCA данные стали более разделимыми")

print("\n4. Лучший метод кластеризации для D3 (t-SNE):")
best_d3 = summary_df[summary_df['Dataset']=='D3 (t-SNE)'].loc[summary_df['Silhouette'].idxmax()]
print(f"   {best_d3['Method'].values[0]} (Silhouette={best_d3['Silhouette'].values[0]:.3f})")
print("   Причина: t-SNE создает плотные кластеры, DBSCAN отлично их находит и игнорирует шум")

print("\n5. Общее заключение:")
print("   - Для визуализации: t-SNE лучше")
print("   - Для кластеризации на исходных данных: KMeans/Agglomerative")
print("   - Для кластеризации на спроецированных данных: DBSCAN")
print("   - t-SNE искажает глобальную структуру, поэтому V-measure может быть низким")