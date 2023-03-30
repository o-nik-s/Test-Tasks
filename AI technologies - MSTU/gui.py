from tkinter import *
from tkinter import ttk

from program import program
import constant as cnst


def gui():
    

    def select_class():
        header.config(text=f"Выбран {selected_class.get()}")
        if selected_class.get() == "Component class":
            forecast_check.set(1)
            forecast_checkbutton.config(state="normal")
        else: forecast_checkbutton.state(["disabled"])


    def build_model_button():
        label["text"] = f"Идет построение моделей на основании {selected_class.get()}"
        btn["state"] = ["disabled"]
        root.update()
        target_column = "10" if selected_class.get() == "Instance class" else "1"
        program(target_column,  forecast_check.get())
        text = f"Модели успешно построены. \nРезультаты тестирования моделей \nсмотрите в файле {cnst.testing_result_file}.\n" + \
            f"Результаты предсказаний смотрите в файлах {cnst.prediction_file} \nи {cnst.prediction_another_file}"
        label["text"] = text
        btn["state"] = ["enabled"]


    root = Tk()
    root.title("Построение модели анализа данных")
    root.geometry("350x230+400+150")

    position = {"padx":6, "pady":6, "anchor":NW}

    header = ttk.Label(text="Выберите класс:")
    header.pack(**position)

    classes = ["Component class", "Instance class"]

    selected_class = StringVar(value=classes[0])

    for clss in classes:
        class_btn = ttk.Radiobutton(text=clss, value=clss, variable=selected_class, command=select_class)
        class_btn.pack(**position)

    forecast_check = IntVar()
 
    forecast_checkbutton = ttk.Checkbutton(text="Построить прогноз для отдельного набора", variable=forecast_check)
    forecast_checkbutton.pack(padx=6, pady=6, anchor=NW)

    label = ttk.Label(text="")
    label.pack()

    btn = ttk.Button(text="Построить модель", command=build_model_button)
    btn.pack()

    root.mainloop()


if __name__ == '__main__':
    gui()