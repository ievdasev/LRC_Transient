# -*- coding: utf-8 -*-
"""
Переходный процесс при замыкании ключа S: оконная программа.
Окно ввода параметров -> расчёт -> таблица результатов и графики.
Расчётное ядро - модуль lrc_transient.py (лежит рядом).
"""
import csv
import multiprocessing as mp
import os
import queue
import threading
from concurrent.futures import ProcessPoolExecutor

import lrc_transient as core

FIELDS = [
    ("Амплитуда напряжения Um, В", "um", "220"),
    ("R секции 1, Ом", "r1", "0.6"),
    ("XL секции 1, Ом", "xl1", "0.2"),
    ("R секций 2…n, Ом", "r2", "0.01"),
    ("XL секций 2…n, Ом", "xl2", "0.003"),
    ("C каждой секции, мкФ", "c", "10"),
    ("Угловая частота ω, рад/с", "w", "314"),
    ("Начальная фаза, град", "phase", "90"),
    ("Макс. время расчёта, мс", "tmax", "60"),
]
DEFAULT_SECTIONS = "1 2 3 5 10 20 30 50"


def num(text):
    """Число с точкой или запятой."""
    return float(str(text).strip().replace(",", "."))


def parse_inputs(values, sections_text):
    """values: {ключ: строка}. Возвращает (p, список_секций) или бросает ValueError с понятным текстом."""
    names = {k: label for label, k, _ in FIELDS}
    v = {}
    for k, s in values.items():
        try:
            v[k] = num(s)
        except ValueError:
            raise ValueError(f"Поле «{names[k]}»: нужно число, введено «{s}»")
    for k in ("um", "c", "w", "tmax"):
        if v[k] <= 0:
            raise ValueError(f"Поле «{names[k]}» должно быть больше нуля")
    for k in ("r1", "xl1", "r2", "xl2"):
        if v[k] < 0:
            raise ValueError(f"Поле «{names[k]}» не может быть отрицательным")
    if v["xl1"] == 0 or v["xl2"] == 0:
        raise ValueError("XL не должно быть равно нулю (нужна индуктивность для расчёта)")
    try:
        secs = sorted({int(x) for x in sections_text.replace(",", " ").split()})
    except ValueError:
        raise ValueError("Секции: введите целые числа через пробел, например «1 2 5 10 50»")
    if not secs or secs[0] < 1:
        raise ValueError("Число секций должно быть не меньше 1")
    if secs[-1] > 300:
        raise ValueError("Слишком много секций (максимум 300)")
    p = dict(um=v["um"], r1=v["r1"], xl1=v["xl1"], r2=v["r2"], xl2=v["xl2"],
             c=v["c"] * 1e-6, w=v["w"], phase=v["phase"], tmax=v["tmax"] * 1e-3)
    return p, secs


def compute(secs, p):
    jobs = [(n, p) for n in secs]
    try:
        with ProcessPoolExecutor(max_workers=max(1, min(len(jobs), os.cpu_count() or 1))) as ex:
            results = list(ex.map(core.run_one, jobs))
    except Exception:                       # если процессы недоступны - считаем последовательно
        results = [core.run_one(j) for j in jobs]
    results.sort(key=lambda r: r[0])
    return results


def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure

    root = tk.Tk()
    root.title("Переходный процесс при замыкании ключа S")
    root.geometry("1280x800")
    root.minsize(1000, 640)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)

    state = {"results": None, "p": None}
    q = queue.Queue()
    vars_ = {}

    # ---------------- левая панель: ввод ----------------
    left = ttk.LabelFrame(root, text=" Входные параметры ", padding=10)
    left.grid(row=0, column=0, sticky="ns", padx=8, pady=8)
    for r, (label, key, default) in enumerate(FIELDS):
        ttk.Label(left, text=label).grid(row=r, column=0, sticky="w", pady=3)
        vars_[key] = tk.StringVar(value=default)
        ttk.Entry(left, textvariable=vars_[key], width=12, justify="right").grid(row=r, column=1, padx=(8, 0))
    r = len(FIELDS)
    ttk.Label(left, text="Числа секций\n(через пробел)").grid(row=r, column=0, sticky="w", pady=(10, 3))
    sec_var = tk.StringVar(value=DEFAULT_SECTIONS)
    ttk.Entry(left, textvariable=sec_var, width=24).grid(row=r + 1, column=0, columnspan=2, sticky="we")
    ttk.Label(left, text="Секция 1: R1, XL1.\nСекции 2…n: R2, XL2.\nXL задаётся на частоте ω.",
              foreground="#555").grid(row=r + 2, column=0, columnspan=2, sticky="w", pady=8)

    btn_calc = ttk.Button(left, text="Рассчитать")
    btn_calc.grid(row=r + 3, column=0, columnspan=2, sticky="we", pady=(6, 3))
    btn_png = ttk.Button(left, text="Сохранить график (PNG)")
    btn_png.grid(row=r + 4, column=0, columnspan=2, sticky="we", pady=3)
    btn_csv = ttk.Button(left, text="Сохранить таблицу (CSV)")
    btn_csv.grid(row=r + 5, column=0, columnspan=2, sticky="we", pady=3)
    btn_def = ttk.Button(left, text="Значения по умолчанию")
    btn_def.grid(row=r + 6, column=0, columnspan=2, sticky="we", pady=3)
    progress = ttk.Progressbar(left, mode="indeterminate")
    progress.grid(row=r + 7, column=0, columnspan=2, sticky="we", pady=(10, 2))
    status = tk.StringVar(value="Введите параметры и нажмите «Рассчитать»")
    ttk.Label(left, textvariable=status, wraplength=250, foreground="#333").grid(
        row=r + 8, column=0, columnspan=2, sticky="w")

    # ---------------- правая панель: графики и таблица ----------------
    right = ttk.Frame(root)
    right.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)
    right.columnconfigure(0, weight=1)
    right.rowconfigure(0, weight=3)
    right.rowconfigure(1, weight=1)

    nb = ttk.Notebook(right)
    nb.grid(row=0, column=0, sticky="nsew")
    figs, canvases = [], []
    for title in ("Ток через ключ S", "Напряжение на конденсаторах"):
        tab = ttk.Frame(nb)
        nb.add(tab, text=title)
        fig = Figure(figsize=(8, 4.5), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=tab)
        NavigationToolbar2Tk(canvas, tab).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        figs.append(fig)
        canvases.append(canvas)

    tbl_frame = ttk.LabelFrame(right, text=" Результаты ", padding=4)
    tbl_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
    tbl_frame.columnconfigure(0, weight=1)
    tbl_frame.rowconfigure(0, weight=1)
    tree = ttk.Treeview(tbl_frame, columns=core.HEAD, show="headings", height=9)
    for h in core.HEAD:
        tree.heading(h, text=h)
        tree.column(h, width=110, anchor="center")
    tree.grid(row=0, column=0, sticky="nsew")
    sb = ttk.Scrollbar(tbl_frame, orient="vertical", command=tree.yview)
    sb.grid(row=0, column=1, sticky="ns")
    tree.configure(yscrollcommand=sb.set)

    # ---------------- действия ----------------
    def set_busy(busy):
        for b in (btn_calc, btn_png, btn_csv, btn_def):
            b.state(["disabled"] if busy else ["!disabled"])
        if busy:
            progress.start(12)
        else:
            progress.stop()

    def worker(secs, p):
        try:
            q.put(("ok", compute(secs, p), p))
        except Exception as e:                  # noqa: BLE001
            q.put(("err", str(e), None))

    def on_calc():
        if "disabled" in btn_calc.state():
            return                              # расчёт уже идёт
        try:
            p, secs = parse_inputs({k: v.get() for k, v in vars_.items()}, sec_var.get())
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return
        set_busy(True)
        status.set(f"Идёт расчёт для секций: {secs} …")
        threading.Thread(target=worker, args=(secs, p), daemon=True).start()
        root.after(150, poll)

    def poll():
        try:
            kind, res, p = q.get_nowait()
        except queue.Empty:
            root.after(150, poll)
            return
        set_busy(False)
        if kind == "err":
            status.set("Ошибка расчёта")
            messagebox.showerror("Ошибка расчёта", res)
            return
        state["results"], state["p"] = res, p
        show(res, p)
        status.set("Готово. Графики можно масштабировать панелью инструментов.")

    def show(res, p):
        tree.delete(*tree.get_children())
        for *_, m in res:
            tree.insert("", "end", values=core.row(m))
        for fig, canvas, drawer in zip(figs, canvases, (core.draw_current, core.draw_voltage)):
            fig.clear()
            drawer(fig.add_subplot(111), res, p)
            fig.tight_layout()
            canvas.draw()

    def on_png():
        if not state["results"]:
            messagebox.showinfo("Нет данных", "Сначала выполните расчёт.")
            return
        idx = nb.index(nb.select())
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")],
                                            initialfile=("ток.png", "напряжение.png")[idx])
        if path:
            figs[idx].savefig(path, dpi=150)
            status.set(f"Сохранено: {path}")

    def on_csv():
        if not state["results"]:
            messagebox.showinfo("Нет данных", "Сначала выполните расчёт.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="результаты.csv")
        if path:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f, delimiter=";")
                w.writerow(core.HEAD)
                w.writerows(core.row(m) for *_, m in state["results"])
            status.set(f"Сохранено: {path}")

    def on_default():
        for _, key, default in FIELDS:
            vars_[key].set(default)
        sec_var.set(DEFAULT_SECTIONS)

    btn_calc.configure(command=on_calc)
    btn_png.configure(command=on_png)
    btn_csv.configure(command=on_csv)
    btn_def.configure(command=on_default)
    root.bind("<Return>", lambda e: on_calc())
    root.mainloop()


if __name__ == "__main__":
    mp.freeze_support()          # обязательно для exe на Windows (процессы расчёта)
    run_gui()
