# Face Verification Streamlit App

Приложение для сравнения лиц на двух фотографиях с помощью нейросети (PyTorch).

## Как запустить

1. Клонируйте репозиторий:
   ```
   git clone https://github.com/Fugel7/data_analysis.git
   cd data_analysis
   ```
2. Установите зависимости:
   ```
   pip install -r requirements.txt
   ```
3. Поместите файл модели `face_verifier_model.pth` в эту папку (если его нет).
4. Запустите приложение:
   ```
   streamlit run face_verifier_app.py
   ```

## Использование
- Загрузите две фотографии.
- Нажмите "Сравнить лица".
- Получите результат схожести и вердикт. 