Проверено с Python 3.13.7. Версии зависимостей: requirements.txt.

Из папки wine_app:
python -m pip install -r requirements.txt
python -m streamlit run app.py

Модель уже обучена; приложение её не обучает. model.joblib содержит StandardScaler и LogisticRegression. Порог 0.2431963642 выбран по максимальной precision при recall >= 0.7 на валидации. На тесте precision=0.4370, recall=0.6789, F1=0.5317. Целевой recall на тесте не достигнут. Данные и split соответствуют исходному заданию. test.csv содержит 1950 вин; example.csv - пример загрузки без target; metrics.csv - сравнение моделей.

st.cache_resource хранит один объект модели для всех сессий и повторных запусков. st.cache_data кэширует таблицы и вероятности, возвращая копии данных. При движении слайдера тестовые вероятности берутся из кэша; меняются метки и метрики. Без декоратора joblib.load перечитывал бы модель при каждом действии.

Для деплоя загрузите папку в GitHub и в share.streamlit.io выберите Create app, репозиторий, ветку и путь wine_app/app.py (или app.py, если загружено только содержимое папки). В Advanced settings укажите Python 3.13, затем Deploy.
Репозиторий: https://github.com/Kurban4ik/mmda-hw3-wine . Точка входа для Streamlit Cloud: app.py; Python 3.13.

Если установка scikit-learn запускает сборку .tar.gz, проверьте версию Python. Для закреплённого scikit-learn==1.6.1 нет готового пакета под Python 3.14 на Linux. Выберите Python 3.13 при деплое. В requirements.txt запрещена сборка из исходников: неподходящее окружение теперь завершится понятной ошибкой вместо долгой компиляции.
