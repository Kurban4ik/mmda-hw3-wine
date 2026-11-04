from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import streamlit as st


BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='Подбор вина', page_icon='W', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(BASE / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(BASE / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def batch_predictions(data, model, features, threshold):
    if data.columns.duplicated().any():
        raise ValueError('В CSV есть повторяющиеся названия столбцов.')
    missing = [name for name in features if name not in data.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if data.empty:
        raise ValueError('В CSV нет строк с данными.')
    if 'target' in data.columns:
        raise ValueError('Загрузите признаки без столбца target.')
    try:
        X = data.loc[:, features].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError):
        raise ValueError('Все признаки должны быть числами.') from None
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('В признаках есть пропуски или бесконечные значения.')
    if not X['is_red'].isin([0, 1]).all():
        raise ValueError('is_red должен содержать только 0 (белое) и 1 (красное).')
    if (X.drop(columns='is_red') < 0).any().any():
        raise ValueError('Физико-химические признаки не могут быть отрицательными.')
    probability = model.predict_proba(X)[:, 1]
    return data.assign(probability=probability, prediction=(probability >= threshold).astype(int))


model = load_model()
test = load_test()
config = json.loads((BASE / 'config.json').read_text(encoding='utf-8'))
features = list(model.feature_names_in_)
X_test = test.loc[:, features]
y_test = test['target'].to_numpy()
p_test = test_proba(model, X_test)

st.title('Подбор вина')
st.caption('Хорошее вино - экспертная оценка не ниже 7. Вероятность - оценка модели.')
threshold = st.sidebar.slider('Порог рекомендации', min_value=0.0, max_value=1.0,
                              value=float(config['threshold']), step=0.001,
                              format='%.3f', key='threshold')
st.sidebar.caption('Порог выбран на валидации. Меняйте его, чтобы сравнить ширину подборки.')
single_tab, metrics_tab, batch_tab, weights_tab = st.tabs([
    'Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

labels = {
    'fixed_acidity': 'Фиксированная кислотность',
    'volatile_acidity': 'Летучая кислотность',
    'citric_acid': 'Лимонная кислота',
    'residual_sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free_sulfur_dioxide': 'Свободный SO2',
    'total_sulfur_dioxide': 'Общий SO2',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь',
    'is_red': 'Красное вино'
}

with single_tab:
    values = {}
    columns = st.columns(3)
    for index, feature in enumerate(features):
        with columns[index % 3]:
            if feature == 'is_red':
                values[feature] = int(st.toggle(labels[feature], value=bool(X_test[feature].median())))
            else:
                values[feature] = st.number_input(
                    labels[feature], min_value=float(X_test[feature].min()),
                    max_value=float(X_test[feature].max()), value=float(X_test[feature].median()),
                    step=0.0001 if feature == 'density' else 0.01,
                    format='%.4f' if feature == 'density' else '%.2f')
    probability = float(model.predict_proba(pd.DataFrame([values], columns=features))[0, 1])
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    st.write('Рекомендовать' if probability >= threshold else 'Не рекомендовать')

with metrics_tab:
    predictions = (p_test >= threshold).astype(int)
    precision = metrics.precision_score(y_test, predictions, zero_division=0)
    recall = metrics.recall_score(y_test, predictions, zero_division=0)
    f1 = metrics.f1_score(y_test, predictions, zero_division=0)
    for column, name, value in zip(st.columns(3), ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
        column.metric(name, f'{value:.3f}')
    st.caption(f'Тест: {len(y_test)} вин. ROC-AUC: {metrics.roc_auc_score(y_test, p_test):.3f}; '
               f'PR-AUC (AP): {metrics.average_precision_score(y_test, p_test):.3f}.')
    cm = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])
    st.dataframe(pd.DataFrame(cm, index=['На самом деле обычное', 'На самом деле хорошее'],
                             columns=['Не рекомендовано', 'Рекомендовано']))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    metrics.RocCurveDisplay.from_predictions(y_test, p_test, ax=axes[0], name='Модель')
    tn, fp, fn, tp = cm.ravel()
    axes[0].scatter(fp / (fp + tn), recall, color='red', zorder=5, label=f'Порог {threshold:.3f}')
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    metrics.PrecisionRecallDisplay.from_predictions(y_test, p_test, ax=axes[1], name='Модель')
    axes[1].scatter(recall, precision, color='red', zorder=5, label=f'Порог {threshold:.3f}')
    axes[1].axhline(y_test.mean(), ls='--', color='gray', label='Случайная подборка')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('Выше порог - уже подборка и обычно выше доля хороших вин, но больше пропусков. '
             'Ниже порог - больше рекомендаций и выше полнота, но больше посредственных вин. '
             'Precision на конечной выборке может колебаться.')

with batch_tab:
    st.write('CSV с 12 признаками, без target. Решение использует общий порог из боковой панели.')
    st.download_button('Скачать пример CSV', data=(BASE / 'example.csv').read_bytes(),
                       file_name='example.csv', mime='text/csv')
    upload = st.file_uploader('Загрузить CSV', type=['csv'])
    if upload is not None:
        try:
            data = pd.read_csv(upload)
            result = batch_predictions(data, model, features, threshold)
        except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as error:
            st.error(str(error))
        else:
            st.dataframe(result)
            st.download_button('Скачать предсказания', result.to_csv(index=False).encode('utf-8-sig'),
                               file_name='wine_predictions.csv', mime='text/csv')

with weights_tab:
    weights = pd.Series(model.named_steps['logisticregression'].coef_[0], index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([labels[name] for name in weights.index], weights.values,
            color=np.where(weights.values >= 0, '#287d53', '#b74747'))
    ax.axvline(0, color='gray', lw=1)
    ax.set_xlabel('Коэффициент после StandardScaler')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    positive = weights[weights > 0].nlargest(3)
    negative = weights[weights < 0].nsmallest(3)
    st.write('За хорошее вино: ' + ', '.join(labels[name] for name in positive.index) + '.')
    st.write('Против: ' + ', '.join(labels[name] for name in negative.index) + '.')
    st.caption('Коэффициенты описывают условные связи при остальных фиксированных признаках; '
               'это не причинные эффекты. Масштаб - одно стандартное отклонение.')
