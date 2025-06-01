import gradio as gr

from pages.rubric_page import rubric_page
from pages.bankrupt_page import bankrupt_page
from pages.pca import pca_page

with gr.Blocks() as demo:
    gr.Markdown("# 📚 ML штучки")

    with gr.Tabs():
        with gr.Tab("🏠 Главная"):
            gr.Markdown("## Добро пожаловать!\nЭто MK штучки:\n- PCA визуализация\n- Рубрикация текста\n - Предсказание банкроства")

        with gr.Tab("🧠 Рубрикация текста"):
            gr.Markdown("### Загрузите `.txt` файл с текстом статьи:")
            txt_file = gr.File(label="Текстовый файл", file_types=[".txt"])
            txt_btn = gr.Button("🔍 Предсказать рубрику")
            txt_output = gr.Markdown()
            txt_btn.click(
                fn=rubric_page, inputs=txt_file, outputs=txt_output
            )
        with gr.Tab("🏦 Предсказание банкротства"):
            gr.Markdown("### Загрузите CSV-файл для классификации банкротства:")
            bankrupt_input = gr.File(label="CSV файл", file_types=[".csv"])
            bankrupt_btn = gr.Button("🔍 Классифицировать")
            bankrupt_output = gr.Dataframe()

            bankrupt_btn.click(
                fn=bankrupt_page,
                inputs=bankrupt_input,
                outputs=[bankrupt_output],
            )
        with gr.Tab("🔬 PCA анализ"):
            gr.Markdown("### Загрузите CSV-файл для PCA анализа")
            pca_input = gr.File(label="CSV файл", file_types=[".csv"])
            pca_button = gr.Button("Построить график")
            pca_output = gr.Plot()

            pca_button.click(fn=pca_page, inputs=pca_input, outputs=pca_output)


if __name__ == "__main__":
    demo.launch()