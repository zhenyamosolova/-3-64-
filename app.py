# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_iris, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(page_title="ML Analysis App", layout="wide")
st.title("🤖 Анализ данных с помощью моделей машинного обучения")

# Боковая панель для выбора параметров
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Выбор датасета
    dataset_name = st.selectbox(
        "Выберите датасет:",
        ("Breast Cancer (классификация)", "Iris (классификация)", "Diabetes (регрессия)")
    )
    
    # Выбор модели в зависимости от типа задачи
    if "классификация" in dataset_name:
        model_type = "classification"
        models = {
            "Логистическая регрессия": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(probability=True, random_state=42)
        }
    else:
        model_type = "regression"
        models = {
            "Линейная регрессия": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "SVR": SVR()
        }
    
    selected_models = st.multiselect(
        "Выберите модели для обучения:",
        options=list(models.keys()),
        default=list(models.keys())[:2]
    )
    
    test_size = st.slider("Размер тестовой выборки:", 0.2, 0.4, 0.25, 0.05)
    random_state = st.number_input("Random state:", 0, 100, 42)

# Загрузка данных
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Breast Cancer (классификация)":
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, data.target_names, model_type
    elif dataset_name == "Iris (классификация)":
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, data.target_names, model_type
    else:  # Diabetes
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df, "regression", model_type

df, target_info, task_type = load_data(dataset_name)

# Основной контент
tab1, tab2, tab3, tab4 = st.tabs(["📊 Данные", "🎯 Обучение", "📈 Результаты", "🔍 Сравнение моделей"])

with tab1:
    st.header("Просмотр данных")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Первые строки данных")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Информация о датасете")
        st.write(f"**Количество строк:** {df.shape[0]}")
        st.write(f"**Количество признаков:** {df.shape[1] - 1}")
        st.write(f"**Тип задачи:** {task_type}")
        if task_type == "classification":
            st.write(f"**Классы:** {target_info}")
            st.write("**Распределение классов:**")
            st.bar_chart(df['target'].value_counts())
        else:
            st.write("**Статистика целевой переменной:**")
            st.write(df['target'].describe())
    
    st.subheader("Статистика признаков")
    st.dataframe(df.describe())
    
    # Визуализация данных
    st.subheader("Корреляционная матрица")
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df.drop('target', axis=1).corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

with tab2:
    st.header("Обучение моделей")
    
    if not selected_models:
        st.warning("Пожалуйста, выберите хотя бы одну модель в боковой панели")
    else:
        # Подготовка данных
        X = df.drop('target', axis=1)
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if task_type == "classification" else None
        )
        
        # Масштабирование
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Кнопка обучения
        if st.button("🚀 Начать обучение", type="primary"):
            results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, model_name in enumerate(selected_models):
                status_text.text(f"Обучение модели: {model_name}...")
                
                model = models[model_name]
                model.fit(X_train_scaled, y_train)
                
                # Предсказания
                y_pred = model.predict(X_test_scaled)
                
                # Метрики
                if task_type == "classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    results[model_name] = {
                        "model": model,
                        "accuracy": accuracy,
                        "cv_mean": cv_scores.mean(),
                        "cv_std": cv_scores.std(),
                        "y_pred": y_pred,
                        "y_test": y_test
                    }
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                    results[model_name] = {
                        "model": model,
                        "mse": mse,
                        "r2": r2,
                        "cv_mean": cv_scores.mean(),
                        "cv_std": cv_scores.std(),
                        "y_pred": y_pred,
                        "y_test": y_test
                    }
                
                progress_bar.progress((i + 1) / len(selected_models))
            
            status_text.text("Обучение завершено!")
            st.session_state['results'] = results
            st.success("✅ Все модели успешно обучены!")
            
            # Отображение метрик
            st.subheader("Метрики качества")
            metrics_df = pd.DataFrame()
            for model_name, res in results.items():
                if task_type == "classification":
                    metrics_df[model_name] = {
                        "Accuracy": f"{res['accuracy']:.4f}",
                        "CV Score (mean)": f"{res['cv_mean']:.4f}",
                        "CV Score (std)": f"{res['cv_std']:.4f}"
                    }
                else:
                    metrics_df[model_name] = {
                        "R² Score": f"{res['r2']:.4f}",
                        "MSE": f"{res['mse']:.2f}",
                        "CV Score (mean)": f"{res['cv_mean']:.4f}",
                        "CV Score (std)": f"{res['cv_std']:.4f}"
                    }
            
            st.dataframe(metrics_df.T)

with tab3:
    st.header("Результаты обучения")
    
    if 'results' not in st.session_state:
        st.info("Сначала обучите модели на вкладке 'Обучение'")
    else:
        results = st.session_state['results']
        
        # Выбор модели для просмотра
        selected_view_model = st.selectbox("Выберите модель для детального просмотра:", list(results.keys()))
        
        if selected_view_model:
            res = results[selected_view_model]
            
            col1, col2 = st.columns(2)
            
            if task_type == "classification":
                with col1:
                    st.metric("Accuracy", f"{res['accuracy']:.4f}")
                    st.metric("CV Score (mean ± std)", f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}")
                    
                    st.subheader("Classification Report")
                    report = classification_report(res['y_test'], res['y_pred'], output_dict=True)
                    st.dataframe(pd.DataFrame(report).T)
                
                with col2:
                    # Confusion Matrix
                    st.subheader("Матрица ошибок")
                    cm = confusion_matrix(res['y_test'], res['y_pred'])
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
                
                # ROC Curve (для бинарной классификации)
                if len(np.unique(res['y_test'])) == 2:
                    st.subheader("ROC Curve")
                    if hasattr(res['model'], "predict_proba"):
                        y_proba = res['model'].predict_proba(res['y_test'].reshape(-1, 1) if len(res['y_test'].shape) == 1 else res['y_test'])[:, 1]
                        fpr, tpr, _ = roc_curve(res['y_test'], y_proba)
                        roc_auc = auc(fpr, tpr)
                        
                        fig, ax = plt.subplots(figsize=(6, 5))
                        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                        ax.plot([0, 1], [0, 1], 'k--')
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.legend()
                        st.pyplot(fig)
            
            else:  # regression
                with col1:
                    st.metric("R² Score", f"{res['r2']:.4f}")
                    st.metric("MSE", f"{res['mse']:.2f}")
                    st.metric("RMSE", f"{np.sqrt(res['mse']):.2f}")
                    st.metric("CV Score (mean ± std)", f"{res['cv_mean']:.4f} ± {res['cv_std']:.4f}")
                
                with col2:
                    # Scatter plot: Actual vs Predicted
                    st.subheader("Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(res['y_test'], res['y_pred'], alpha=0.6)
                    ax.plot([res['y_test'].min(), res['y_test'].max()], 
                           [res['y_test'].min(), res['y_test'].max()], 'r--', lw=2)
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    st.pyplot(fig)
                
                # Residuals plot
                st.subheader("График остатков")
                residuals = res['y_test'] - res['y_pred']
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.scatter(res['y_pred'], residuals, alpha=0.6)
                ax.axhline(y=0, color='r', linestyle='--')
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Residuals")
                st.pyplot(fig)

with tab4:
    st.header("Сравнение моделей")
    
    if 'results' not in st.session_state:
        st.info("Сначала обучите модели на вкладке 'Обучение'")
    else:
        results = st.session_state['results']
        
        # Сравнительные графики
        if task_type == "classification":
            metrics_to_plot = ['accuracy', 'cv_mean']
            metric_names = ['Accuracy', 'Cross-validation Score']
        else:
            metrics_to_plot = ['r2', 'cv_mean']
            metric_names = ['R² Score', 'Cross-validation Score']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for idx, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
            values = [results[model][metric] for model in results.keys()]
            bars = axes[idx].bar(results.keys(), values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(results)])
            axes[idx].set_title(metric_name)
            axes[idx].set_ylabel(metric_name)
            axes[idx].set_ylim([min(values) - 0.1, max(values) + 0.1])
            
            # Добавление значений на столбцы
            for bar, val in zip(bars, values):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                              f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Таблица сравнения
        st.subheader("Детальное сравнение")
        comparison_df = pd.DataFrame()
        for model_name, res in results.items():
            if task_type == "classification":
                comparison_df[model_name] = {
                    "Accuracy": f"{res['accuracy']:.4f}",
                    "CV Mean": f"{res['cv_mean']:.4f}",
                    "CV Std": f"{res['cv_std']:.4f}"
                }
            else:
                comparison_df[model_name] = {
                    "R² Score": f"{res['r2']:.4f}",
                    "MSE": f"{res['mse']:.2f}",
                    "RMSE": f"{np.sqrt(res['mse']):.2f}",
                    "CV Mean": f"{res['cv_mean']:.4f}",
                    "CV Std": f"{res['cv_std']:.4f}"
                }
        
        st.dataframe(comparison_df)
        
        # Лучшая модель
        if task_type == "classification":
            best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
            st.success(f"🏆 Лучшая модель по accuracy: **{best_model[0]}** (accuracy: {best_model[1]['accuracy']:.4f})")
        else:
            best_model = max(results.items(), key=lambda x: x[1]['r2'])
            st.success(f"🏆 Лучшая модель по R²: **{best_model[0]}** (R²: {best_model[1]['r2']:.4f})")