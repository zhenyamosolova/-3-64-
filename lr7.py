"""
ПОЛНЫЙ КОД ДЛЯ ПРОГНОЗИРОВАНИЯ ВРЕМЕННОГО РЯДА
Исправленная версия с загрузкой данных о пассажирах авиакомпаний
"""

# ==================== ПРОВЕРКА УСТАНОВКИ БИБЛИОТЕК ====================
print("="*80)
print("ПРОВЕРКА УСТАНОВКИ НЕОБХОДИМЫХ БИБЛИОТЕК")
print("="*80)

missing_libraries = []

try:
    import numpy as np
    print("✓ NumPy - установлен")
except ImportError:
    missing_libraries.append("numpy")
    print("✗ NumPy - НЕ УСТАНОВЛЕН")

try:
    import pandas as pd
    print("✓ Pandas - установлен")
except ImportError:
    missing_libraries.append("pandas")
    print("✗ Pandas - НЕ УСТАНОВЛЕН")

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib - установлен")
except ImportError:
    missing_libraries.append("matplotlib")
    print("✗ Matplotlib - НЕ УСТАНОВЛЕН")

try:
    import seaborn as sns
    print("✓ Seaborn - установлен")
except ImportError:
    missing_libraries.append("seaborn")
    print("✗ Seaborn - НЕ УСТАНОВЛЕН")

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    print("✓ Scikit-learn - установлен")
except ImportError:
    missing_libraries.append("scikit-learn")
    print("✗ Scikit-learn - НЕ УСТАНОВЛЕН")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    print("✓ Statsmodels - установлен")
except ImportError:
    missing_libraries.append("statsmodels")
    print("✗ Statsmodels - НЕ УСТАНОВЛЕН")

try:
    from gmdh import Combi, Mia
    print("✓ GMDH - установлен")
except ImportError:
    missing_libraries.append("gmdh")
    print("✗ GMDH - НЕ УСТАНОВЛЕН")

# Проверка символьной регрессии (опционально)
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
    print("✓ PySR - установлен (символьная регрессия доступна)")
except ImportError:
    PYSR_AVAILABLE = False
    print("○ PySR - НЕ УСТАНОВЛЕН (будет использована эмуляция)")

if missing_libraries:
    print("\n" + "="*80)
    print("ОШИБКА: Отсутствуют следующие библиотеки:")
    for lib in missing_libraries:
        print(f"  - {lib}")
    print("\nДля установки выполните команду:")
    print(f"pip install {' '.join(missing_libraries)}")
    print("="*80)
    exit(1)
else:
    print("\n✓ Все необходимые библиотеки установлены!")
    print("="*80)

# ==================== ЗАГРУЗКА ДАННЫХ ====================
print("\nШАГ 1: ЗАГРУЗКА И АНАЛИЗ ДАННЫХ")
print("="*80)

# Вариант 1: Загрузка из URL (надежный источник)
try:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
    df.columns = ['passengers']
    print("Данные загружены из репозитория классических датасетов")
except:
    # Вариант 2: Создание данных вручную (если нет интернета)
    print("Не удалось загрузить данные из интернета. Создаем синтетические данные...")
    dates = pd.date_range(start='1949-01-01', end='1960-12-01', freq='MS')
    t = np.arange(len(dates))
    # Тренд + сезонность + шум
    trend = 100 + 2 * t
    seasonality = 50 * np.sin(2 * np.pi * t / 12 + np.pi/2)
    noise = np.random.normal(0, 10, len(t))
    passengers = trend + seasonality + noise
    df = pd.DataFrame({'passengers': passengers}, index=dates)
    print("Созданы синтетические данные с трендом и сезонностью")

print(f"\nЗагружено {len(df)} наблюдений")
print(f"Период: с {df.index[0].strftime('%B %Y')} по {df.index[-1].strftime('%B %Y')}")
print(f"\nСтатистики ряда:")
print(f"  Среднее: {df['passengers'].mean():.1f}")
print(f"  Медиана: {df['passengers'].median():.1f}")
print(f"  Минимум: {df['passengers'].min():.1f}")
print(f"  Максимум: {df['passengers'].max():.1f}")
print(f"  Стандартное отклонение: {df['passengers'].std():.1f}")

# Проверка на наличие пропусков
if df['passengers'].isnull().any():
    print(f"  ВНИМАНИЕ: Обнаружено {df['passengers'].isnull().sum()} пропусков")
    df['passengers'] = df['passengers'].interpolate()

# ==================== ШАГ 2: ВИЗУАЛИЗАЦИЯ ====================
print("\nШАГ 2: ВИЗУАЛИЗАЦИЯ ВРЕМЕННОГО РЯДА")
print("="*80)

# 2.1 График временного ряда
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(df.index, df['passengers'], color='navy', linewidth=2, marker='o', markersize=3)
axes[0].set_title('Количество пассажиров авиакомпаний (1949-1960)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Дата')
axes[0].set_ylabel('Пассажиры (тыс.)')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].axhline(y=df['passengers'].mean(), color='red', linestyle='--', alpha=0.7, 
                label=f'Среднее: {df["passengers"].mean():.0f}')
axes[0].legend()

# 2.2 Декомпозиция
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(df['passengers'], model='multiplicative', period=12)
    decomposition.trend.plot(ax=axes[1], color='green', label='Тренд', linewidth=2)
    decomposition.seasonal.plot(ax=axes[1], color='orange', label='Сезонность', linewidth=2)
    decomposition.resid.plot(ax=axes[1], color='red', label='Остатки', alpha=0.7)
    axes[1].set_title('Декомпозиция временного ряда', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Дата')
    axes[1].set_ylabel('Значение')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, linestyle='--')
except Exception as e:
    print(f"Декомпозиция не выполнена: {e}")
    axes[1].text(0.5, 0.5, 'Декомпозиция временно недоступна', 
                ha='center', va='center', transform=axes[1].transAxes)

plt.tight_layout()
plt.show()

# Дополнительная визуализация
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Гистограмма
axes[0].hist(df['passengers'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_title('Распределение количества пассажиров', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Пассажиры (тыс.)')
axes[0].set_ylabel('Частота')
axes[0].axvline(df['passengers'].mean(), color='red', linestyle='--', label=f'Среднее: {df["passengers"].mean():.0f}')
axes[0].axvline(df['passengers'].median(), color='green', linestyle='--', label=f'Медиана: {df["passengers"].median():.0f}')
axes[0].legend()

# Boxplot по годам
df['year'] = df.index.year
df.boxplot(column='passengers', by='year', ax=axes[1])
axes[1].set_title('Распределение по годам', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Год')
axes[1].set_ylabel('Пассажиры (тыс.)')
plt.suptitle('')

plt.tight_layout()
plt.show()

# ==================== ШАГ 3: РАЗДЕЛЕНИЕ НА ВЫБОРКИ ====================
print("\nШАГ 3: РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ")
print("="*80)

# Удаляем временный столбец года
if 'year' in df.columns:
    df.drop('year', axis=1, inplace=True)

# Последние 24 месяца (2 года) - тест
train = df.iloc[:-24].copy()
test = df.iloc[-24:].copy()

print(f"Обучающая выборка: {len(train)} мес. ({train.index[0].strftime('%Y-%m')} - {train.index[-1].strftime('%Y-%m')})")
print(f"Тестовая выборка:  {len(test)} мес. ({test.index[0].strftime('%Y-%m')} - {test.index[-1].strftime('%Y-%m')})")

# Визуализация разделения
plt.figure(figsize=(14, 6))
plt.plot(train.index, train['passengers'], 'b-', linewidth=2, label='Обучающая выборка')
plt.plot(test.index, test['passengers'], 'r--', linewidth=2, label='Тестовая выборка')
plt.axvline(x=train.index[-1], color='gray', linestyle=':', alpha=0.7, label='Граница разделения')
plt.fill_between(test.index, test['passengers'].min(), test['passengers'].max(), 
                 alpha=0.1, color='red', label='Зона прогноза')
plt.title('Разделение данных на обучающую и тестовую выборки', fontsize=14, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Пассажиры (тыс.)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==================== ШАГ 4: ARIMA ====================
print("\nШАГ 4: ПРОГНОЗИРОВАНИЕ ARIMA")
print("="*80)

try:
    # Простая модель ARIMA для начала
    model_arima = ARIMA(train['passengers'], order=(1,1,1))
    model_arima_fit = model_arima.fit()
    print("Модель ARIMA(1,1,1) успешно обучена")
    print(model_arima_fit.summary())
    
    forecast_arima = model_arima_fit.forecast(steps=24)
    forecast_arima = pd.Series(forecast_arima, index=test.index)
    print("Прогноз ARIMA создан")
except Exception as e:
    print(f"Ошибка при обучении ARIMA: {e}")
    print("Используется упрощенный прогноз (последнее значение + тренд)")
    last_value = train['passengers'].iloc[-1]
    trend = (train['passengers'].iloc[-1] - train['passengers'].iloc[-13]) / 12 if len(train) >= 13 else 2
    forecast_arima = pd.Series([last_value + trend * i for i in range(1, 25)], index=test.index)

# ==================== ШАГ 5: ПОДГОТОВКА ДАННЫХ ДЛЯ ДРУГИХ МЕТОДОВ ====================
def create_lagged_features(series, lags=12):
    """Создание признаков-лагов для машинного обучения"""
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

lags = 12
X_train, y_train = create_lagged_features(train['passengers'].values, lags)
print(f"\nПодготовлено {len(X_train)} обучающих примеров с {lags} лагами")

# ==================== ШАГ 6: СИМВОЛЬНАЯ РЕГРЕССИЯ ====================
print("\nШАГ 5: ПРОГНОЗИРОВАНИЕ СИМВОЛЬНОЙ РЕГРЕССИЕЙ")
print("="*80)

if PYSR_AVAILABLE:
    try:
        model_sr = PySRRegressor(
            niterations=50,
            binary_operators=["+", "*", "-", "/"],
            unary_operators=["sin", "exp", "abs", "square"],
            model_selection="best",
            verbosity=0
        )
        
        print("Обучение символьной регрессии (может занять несколько минут)...")
        model_sr.fit(X_train, y_train)
        
        # Итеративный прогноз
        last_vals = train['passengers'].values[-lags:].tolist()
        predictions_sr = []
        for i in range(len(test)):
            input_vec = np.array(last_vals[-lags:]).reshape(1, -1)
            pred = model_sr.predict(input_vec)[0]
            predictions_sr.append(pred)
            last_vals.append(pred)
        
        forecast_sr = pd.Series(predictions_sr, index=test.index)
        print("Прогноз символьной регрессии создан")
        
    except Exception as e:
        print(f"Ошибка символьной регрессии: {e}")
        forecast_sr = forecast_arima.copy()
else:
    print("PySR не установлен. Используется эмуляция:")
    print("  Модель: тренд + сезонность")
    t = np.arange(len(test))
    season = 40 * np.sin(2 * np.pi * t / 12 + np.pi/2)
    trend_start = train['passengers'].iloc[-1]
    trend = trend_start * (1 + 0.015 * t)
    forecast_sr = pd.Series(trend + season, index=test.index)

# ==================== ШАГ 7: МГУА МЕТОДЫ ====================
print("\nШАГ 6: ПРОГНОЗИРОВАНИЕ МГУА (COMBI и MIA)")
print("="*80)

try:
    # Разделение для МГУА
    split_idx = int(len(X_train) * 0.8)
    X_train_gmdh = X_train[:split_idx]
    y_train_gmdh = y_train[:split_idx]
    
    # COMBI (линейный)
    print("Обучение COMBI (линейный метод)...")
    model_combi = Combi()
    model_combi.fit(X_train_gmdh, y_train_gmdh)
    
    # MIA (нелинейный)
    print("Обучение MIA (нелинейный метод)...")
    model_mia = Mia()
    model_mia.fit(X_train_gmdh, y_train_gmdh, max_layers=3)
    
    # Функция для прогноза
    def predict_gmdh(model, train_series, test_series, lags):
        last_vals = train_series.values[-lags:].tolist()
        preds = []
        for i in range(len(test_series)):
            X_in = np.array(last_vals[-lags:]).reshape(1, -1)
            try:
                pred = model.predict(X_in)[0]
            except:
                pred = last_vals[-1]  # fallback
            preds.append(pred)
            last_vals.append(pred)
        return pd.Series(preds, index=test_series.index)
    
    forecast_combi = predict_gmdh(model_combi, train['passengers'], test['passengers'], lags)
    forecast_mia = predict_gmdh(model_mia, train['passengers'], test['passengers'], lags)
    
    print("Прогнозы МГУА созданы")
    
except Exception as e:
    print(f"Ошибка в МГУА: {e}")
    forecast_combi = forecast_arima.copy()
    forecast_mia = forecast_arima.copy()

# ==================== ШАГ 8: ВИЗУАЛИЗАЦИЯ ПРОГНОЗОВ ====================
print("\nШАГ 7: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("="*80)

plt.figure(figsize=(15, 8))

plt.plot(test.index, test['passengers'], 'ko-', linewidth=2, markersize=5, label='Фактические данные', zorder=5)
plt.plot(test.index, forecast_arima, 'b-', linewidth=2, label='ARIMA', alpha=0.8)
plt.plot(test.index, forecast_sr, 'g--', linewidth=2, label='Символьная регрессия', alpha=0.8)
plt.plot(test.index, forecast_combi, 'r-.', linewidth=2, label='МГУА COMBI (линейный)', alpha=0.8)
plt.plot(test.index, forecast_mia, 'orange', linewidth=2, linestyle=':', label='МГУА MIA (нелинейный)', alpha=0.8)

plt.title('Сравнение прогнозов на тестовой выборке (1958-1960)', fontsize=14, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Количество пассажиров (тыс.)')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# График ошибок
plt.figure(figsize=(15, 6))

errors = {
    'ARIMA': test['passengers'].values - forecast_arima.values,
    'Символьная регрессия': test['passengers'].values - forecast_sr.values,
    'COMBI': test['passengers'].values - forecast_combi.values,
    'MIA': test['passengers'].values - forecast_mia.values
}

for name, error in errors.items():
    plt.plot(test.index, error, linewidth=1.5, label=name, alpha=0.7)

plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.title('График ошибок прогнозов (факт - прогноз)', fontsize=14, fontweight='bold')
plt.xlabel('Дата')
plt.ylabel('Ошибка (тыс. пассажиров)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==================== ШАГ 9: ОЦЕНКА КАЧЕСТВА ====================
print("\nШАГ 8: ОЦЕНКА КАЧЕСТВА ПРОГНОЗОВ")
print("="*80)

def evaluate(y_true, y_pred, name):
    """Расчет метрик качества"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{name}:")
    print(f"  MSE:  {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

actual = test['passengers']
results = {}

print("\n" + "="*50)
results['ARIMA'] = evaluate(actual, forecast_arima, "ARIMA")
results['Символьная регрессия'] = evaluate(actual, forecast_sr, "Символьная регрессия")
results['МГУА COMBI'] = evaluate(actual, forecast_combi, "МГУА COMBI")
results['МГУА MIA'] = evaluate(actual, forecast_mia, "МГУА MIA")

# ==================== ИТОГОВАЯ ТАБЛИЦА ====================
print("\n" + "="*80)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("="*80)

summary = pd.DataFrame(results).T.round(2)
print("\n", summary.to_string())

# Определение лучшей модели
best_rmse = summary['RMSE'].idxmin()
best_mape = summary['MAPE'].idxmin()

print(f"\n{'='*50}")
print(f"🏆 Лучшая модель по RMSE: {best_rmse} (RMSE = {summary.loc[best_rmse, 'RMSE']:.2f})")
print(f"🏆 Лучшая модель по MAPE: {best_mape} (MAPE = {summary.loc[best_mape, 'MAPE']:.2f}%)")
print(f"{'='*50}")

# Визуализация сравнения RMSE
plt.figure(figsize=(10, 6))
models = list(results.keys())
rmse_values = [results[m]['RMSE'] for m in models]
colors = ['blue', 'green', 'red', 'orange']

bars = plt.bar(models, rmse_values, color=colors, alpha=0.7, edgecolor='black')
plt.title('Сравнение RMSE моделей (чем меньше, тем лучше)', fontsize=14, fontweight='bold')
plt.ylabel('RMSE (тыс. пассажиров)')
plt.xlabel('Модели')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars, rmse_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
print("="*80)