import matplotlib.pyplot as plt

from service.models.pca.pca import PCA



def pca_page(file_path):

    if file_path is  None:
        return "Нет файла"
    try:
        pca = PCA(file_path)
        x_scalled, x_pca, y1 = pca.transform()
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(x_pca[:, 0], x_pca[:, 1], c=y1, cmap='viridis')
        fig.colorbar(scatter, ax=ax).set_label('')
        ax.set_xlabel('Первая главная компонента')
        ax.set_ylabel('Вторая главная компонента')
        ax.set_title('Результаты PCA для набора данных Rural_prices')
        return fig
    except Exception as e:
        return f"Ошибка при обработке файла: {e}"
