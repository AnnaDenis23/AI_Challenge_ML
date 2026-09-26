# 💡 Light Level Classifier

**AI Challenge 2026** · Computer Vision · ResNet18

MVP-приложение на Streamlit для определения уровня освещённости на изображениях с камер видеонаблюдения.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B)

---

## 🎯 Задача

Классификация изображений по уровню освещённости на 3 класса:

| Класс | Название | Описание |
|-------|----------|----------|
| `0` | 🌑 **dark** | Низкая освещённость |
| `1` | 🌤️ **normal** | Средняя освещённость |
| `2` | ☀️ **bright** | Высокая освещённость |

**Метрика:** Accuracy  
**Результат:** 0.52

---

## 🧠 Подход

- **Архитектура:** ResNet18 (ImageNet pretrained)
- **Transfer Learning:** заморожен backbone, обучены `layer4` + кастомная голова
- **Аугментации:** RandomHorizontalFlip, RandomRotation, RandomAffine, ColorJitter
- **Оптимизатор:** Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler:** ReduceLROnPlateau
- **Early Stopping:** patience=5
- **TTA:** усреднение по 5 аугментациям при инференсе

---

## 📁 Структура проекта

```
light-classifier/
├── app/                       # Streamlit-приложение
│   ├── app.py                 # UI
│   ├── model.py               # Архитектура
│   ├── utils.py               # Утилиты и TTA
│   └── best_model.pth         # Веса модели (скачать после обучения)
│
├── colab/
│   └── train_colab.ipynb      # Обучение в Google Colab
│
├── csv/                       # Данные
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Обучение модели в Google Colab

1. Откройте `colab/train_colab.ipynb` в [Google Colab](https://colab.research.google.com)
2. Включите GPU: `Runtime → Change runtime type → T4 GPU`
3. Загрузите датасет на Google Drive
4. Запустите все ячейки
5. Скачайте `best_model.pth` из Google Drive

### 3. Запуск приложения

```bash
# Положите best_model.pth в папку app/
cp ~/Downloads/best_model.pth app/best_model.pth

# Запустите приложение
cd app
streamlit run app.py
```

Приложение откроется на `http://localhost:8501`

---

## 🖼️ Использование

1. Загрузите изображение через интерфейс
2. Приложение покажет:
   - Предсказанный класс (🌑 / 🌤️ / ☀️)
   - Вероятности по каждому классу
   - Уверенность модели
3. Опционально включите TTA для более точных предсказаний

---

## 📊 Результаты

| Класс | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| dark | 0.54 | 0.68 | 0.60 |
| normal | 0.55 | 0.55 | 0.55 |
| bright | 0.45 | 0.33 | 0.38 |

**Overall Accuracy:** 0.52

---

## 📦 Стек

`PyTorch` · `torchvision` · `Streamlit` · `scikit-learn` · `Pandas` · `Pillow`

---

## 📥 Где взять веса модели

Веса (`best_model.pth`, ~45 МБ) не хранятся в репозитории из-за размера. Скачать можно:

- **Google Drive:** [ссылка на ваш файл]
- **GitHub Releases:** [ссылка на релиз]

После скачивания положите файл в папку `app/`.