from service.models.logit.classify_is_bankrupt import ClassifyIsBankrupt


def bankrupt_page(file_path):
    if file_path is None:
        return "Нет файла"

    try:
        c = ClassifyIsBankrupt(file_path)
        df = c.read_file()
        if "Х1" not in df.columns or "Х2" not in df.columns:
            raise ValueError("Колонки 'X1' и 'X2' не найдены после очистки")
        predictions = c.predict()
        df["Предсказание"] = predictions
        return df

    except Exception as e:
        return f"Ошибка при обработке файла: {e}"
