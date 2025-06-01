import os
import uuid

from service.models.knn.text_clasifier import ClassifyText



def rubric_page(uploaded_file):
    if uploaded_file is None:
        return f"Нет файла"

    try:
        c = ClassifyText(uploaded_file)
        return c.classify_text()
    except Exception as e:
        return f"Ошибка при обработке файла: {e}"
    finally:
        if os.path.exists(uploaded_file):
            os.remove(uploaded_file)

