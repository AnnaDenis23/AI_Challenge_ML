import os
import sys
import numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import create_model, CLASS_NAMES, CLASS_NAMES_RU, CLASS_EMOJI, CLASS_COLOR
from utils import predict_single

# ============================================
# КОНФИГУРАЦИЯ
# ============================================
st.set_page_config(
    page_title="Light Level Classifier",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================
# ЗАГРУЗКА МОДЕЛИ (кэшируется)
# ============================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Файл не найден: {MODEL_PATH}"
    try:
        model = create_model(num_classes=3, pretrained=False)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
        return model, f"OK ({size_mb:.1f} MB)"
    except Exception as e:
        return None, f"Ошибка загрузки: {e}"


# ============================================
# ХЕДЕР
# ============================================
st.title("💡 Классификатор уровня освещённости")
st.markdown("""
**AI Challenge 2026** · Computer Vision · ResNet18 (Transfer Learning)

Определяет уровень освещённости на изображениях с камер видеонаблюдения.
""")
st.divider()

# ============================================
# ЗАГРУЗКА МОДЕЛИ И ОБРАБОТКА ОШИБОК
# ============================================
model, status = load_model()

if model is None:
    st.error(f"⚠️ Не удалось загрузить модель: `{status}`")
    st.info("""
    ### 📥 Как получить веса модели
    
    1. Откройте `colab/train_colab.ipynb` в Google Colab
    2. Запустите обучение (Runtime → Change runtime type → **T4 GPU**)
    3. Скачайте файл `best_model.pth`
    4. Поместите его в папку `app/`
    5. Перезапустите приложение: `streamlit run app.py`
    
    Подробнее — в `README.md`.
    """)
    st.stop()

# ============================================
# БОКОВАЯ ПАНЕЛЬ
# ============================================
with st.sidebar:
    st.success(f"✅ Модель загружена: {status}")
    st.divider()

    st.header("⚙️ Настройки инференса")
    use_tta = st.checkbox(
        "Использовать TTA",
        value=True,
        help="Test Time Augmentation — усредняет предсказания по нескольким аугментациям. Точнее, но медленнее."
    )
    n_aug = st.slider("Количество аугментаций", 3, 10, 5) if use_tta else 1

    st.divider()
    st.markdown("### 📊 О модели")
    st.markdown("""
    - **Архитектура:** ResNet18
    - **Pretrained:** ImageNet
    - **Классов:** 3
    - **Метрика:** Accuracy
    - **Test Accuracy:** ~0.52
    """)

    st.divider()
    st.markdown("### 📖 Классы")
    st.markdown("""
    - 🌑 **Тёмно** — низкая освещённость
    - 🌤️ **Нормально** — средняя освещённость
    - ☀️ **Ярко** — высокая освещённость
    """)

    st.divider()
    st.caption(f"Device: `{DEVICE}`")

# ============================================
# ОСНОВНАЯ ОБЛАСТЬ
# ============================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Изображение")
    uploaded_file = st.file_uploader(
        "Загрузите изображение",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Загруженное изображение", use_container_width=True)
    else:
        st.info("👈 Загрузите изображение, чтобы получить предсказание")

with col2:
    st.subheader("🎯 Результат")

    if uploaded_file is not None:
        with st.spinner("Анализирую изображение..."):
            pred_class, probs = predict_single(
                model, image, DEVICE,
                use_tta=use_tta,
                n_augmentations=n_aug
            )

        class_ru = CLASS_NAMES_RU[pred_class]
        class_en = CLASS_NAMES[pred_class]
        emoji = CLASS_EMOJI[pred_class]
        color = CLASS_COLOR[pred_class]

        # Большая плашка с предсказанием
        st.markdown(
            f"""
            <div style="
                background-color: {color};
                padding: 24px;
                border-radius: 14px;
                text-align: center;
                color: white;
                margin-bottom: 20px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.15);
            ">
                <h1 style="margin: 0; font-size: 2.5em;">{emoji} {class_ru}</h1>
                <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 1.1em;">({class_en})</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Вероятности
        st.markdown("#### Вероятности по классам")
        for i, (name_ru, prob) in enumerate(zip(CLASS_NAMES_RU.values(), probs)):
            marker = "🎯 " if i == pred_class else ""
            st.markdown(f"{marker}**{name_ru}** — {prob * 100:.2f}%")
            st.progress(float(prob))

        # Уверенность
        confidence = float(np.max(probs))
        if confidence > 0.8:
            st.success(f"✅ Высокая уверенность: **{confidence * 100:.1f}%**")
        elif confidence > 0.6:
            st.warning(f"⚠️ Средняя уверенность: **{confidence * 100:.1f}%**")
        else:
            st.error(f"❌ Низкая уверенность: **{confidence * 100:.1f}%** — модель не уверена")

# ============================================
# ФУТЕР
# ============================================
st.divider()
st.caption("© 2026 Anastasiya Denisenko · AI Challenge · ResNet18 + PyTorch + Streamlit")