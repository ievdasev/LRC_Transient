#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Переходный процесс при замыкании ключа S в цепочке секций
    источник u(t) -> S -> [R-L -> мост -> C] -> [R-L -> мост -> C] -> ...

Секция 1 имеет параметры R1, XL1, остальные секции (2...n) - R2, XL2.
Все мосты - двухполупериодные идеальные выпрямители, нагрузки на C нет,
нижний провод источника общий для всех мостов.

Источник: u(t) = Um * sin(w*t + phase). По умолчанию Um = 220 В, w = 314 рад/с,
phase = 90 град (ключ замыкается в максимуме напряжения).

Выход: таблица (консоль + CSV) и график (PNG) с током через ключ S для заданных
чисел секций, отмечены максимум тока и уровень Амплитуда/2.

Запуск:
    python lrc_transient.py                 # интерактивный ввод параметров
    python lrc_transient.py --um 220 --r1 0.6 --xl1 0.2 --r2 0.01 --xl2 0.003 --c 10 \
                            --sections 1 2 3 5 10 20 30 50
Нужно: numpy, scipy, matplotlib.
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import solve_banded

GON, GOFF = 1 / 2e-3, 1e-9      # идеальный диод: Ron = 2 мОм, Roff -> бесконечность


# --------------------------------------------------------------------------- #
#                                   РАСЧЁТ                                    #
# --------------------------------------------------------------------------- #
def simulate(n, p):
    """Возвращает (t, i_S, uC[:, n]) для цепочки из n секций."""
    Um, w, ph = p["um"], p["w"], np.radians(p["phase"])
    C = p["c"]
    Rs = np.full(n, p["r2"]); Rs[0] = p["r1"]
    Ls = np.full(n, p["xl2"] / w); Ls[0] = p["xl1"] / w

    dt_fine = min(2e-7, 2 * np.pi * np.sqrt(Ls.min() * C) / 2000)   # шаг во время импульса
    dt_fine = max(dt_fine, 1e-9)
    dt_coarse = max(2e-6, dt_fine)                                    # шаг в паузах
    tmax = p["tmax"]

    idx = np.arange(n)
    ri, rc, rv = 3 * idx, 3 * idx + 1, 3 * idx + 2    # порядок неизвестных: i_k, uC_k, uT_k

    def u(t):
        return Um * np.sin(w * t + ph)

    i = np.zeros(n); vc = np.zeros(n)
    st = np.zeros(n, int)         # 0 - мост закрыт, +1 - открыт (uT > uC), -1 - открыт (uT < -uC)
    ts = [0.0]; I = [0.0]; V = [vc.astype(np.float32)]
    t = 0.0; t_active = 0.0

    while t < tmax:
        active = bool((st != 0).any() or np.abs(i).max() > 1e-4 or t < 20 * dt_fine)
        if active:
            t_active = t
        elif vc.min() >= Um * 1.0001 and t - t_active > 100e-6:
            break                                      # диоды больше не откроются
        dt = dt_fine if active else dt_coarse
        t += dt
        rhs = np.zeros(3 * n)
        rhs[ri] = Ls / dt * i
        rhs[0] += u(t)
        rhs[rc] = C / dt * vc
        for _ in range(40):                            # подбор состояний диодов
            g = np.where(st != 0, GON, GOFF)
            sg = np.where(st == -1, -1.0, 1.0)
            ab = np.zeros((5, 3 * n))                  # ленточная матрица (kl = ku = 2)

            def put(r, c, val):
                ab[2 + r - c, c] = val

            put(ri, ri, Ls / dt + Rs); put(ri, rv, 1.0)
            put(ri[1:], rv[:-1], -1.0)
            put(rc, rc, C / dt + g); put(rc, rv, -sg * g)
            put(rv, ri, 1.0); put(rv[:-1], ri[1:], -1.0)
            put(rv, rv, -g); put(rv, rc, sg * g)
            x = solve_banded((2, 2), ab, rhs)
            ni, nvc, nvt = x[ri], x[rc], x[rv]
            nst = st.copy()
            z = st == 0
            nst[z & (nvt - nvc > 1e-9)] = 1
            nst[z & (-nvt - nvc > 1e-9)] = -1
            nst[(st == 1) & (nvt - nvc < 0)] = 0
            nst[(st == -1) & (-nvt - nvc < 0)] = 0
            if (nst == st).all():
                break
            st = nst
        i, vc = ni, nvc
        ts.append(t); I.append(i[0]); V.append(vc.astype(np.float32))
    return np.array(ts), np.array(I), np.array(V)


def metrics(t, i):
    """Амплитуда, уровень A/2 и времена пересечения (по модулю тока)."""
    a = np.abs(i)
    k = int(np.argmax(a)); A = float(a[k]); h = A / 2
    lo = k
    while lo > 0 and a[lo] > h:
        lo -= 1
    hi = k
    while hi < len(a) - 1 and a[hi] > h:
        hi += 1
    t1 = t[lo] + (h - a[lo]) / (a[lo + 1] - a[lo]) * (t[lo + 1] - t[lo])
    t2 = t[hi - 1] + (h - a[hi - 1]) / (a[hi] - a[hi - 1]) * (t[hi] - t[hi - 1])
    e = k
    while e < len(a) - 1 and a[e] > 1e-3:
        e += 1
    return dict(A=A, tA=t[k], half=h, t1=t1, t2=t2, width=t2 - t1, tend=t[e])


def run_one(args):
    n, p = args
    t, i, v = simulate(n, p)
    m = metrics(t, i)
    m.update(n=n, uc_min=float(v[-1].min()), uc_max=float(v[-1].max()))
    return n, t, i, v, m


# --------------------------------------------------------------------------- #
#                                 ВЫВОД                                       #
# --------------------------------------------------------------------------- #
HEAD = ["Секций", "Imax, А", "t(Imax), мкс", "A/2, А", "t1, мкс", "t2, мкс",
        "Ширина A/2, мкс", "Конец импульса, мкс", "uC конц., В"]


def row(m):
    uc = f"{m['uc_min']:.0f}" if abs(m['uc_max'] - m['uc_min']) < 0.5 else f"{m['uc_min']:.0f}…{m['uc_max']:.0f}"
    return [str(m["n"]), f"{m['A']:.1f}", f"{m['tA']*1e6:.0f}", f"{m['half']:.1f}",
            f"{m['t1']*1e6:.0f}", f"{m['t2']*1e6:.0f}", f"{m['width']*1e6:.0f}",
            f"{m['tend']*1e6:.0f}", uc]


def print_table(rows):
    wd = [max(len(HEAD[c]), *(len(r[c]) for r in rows)) for c in range(len(HEAD))]
    line = " | ".join(h.center(wd[c]) for c, h in enumerate(HEAD))
    print(line); print("-" * len(line))
    for r in rows:
        print(" | ".join(r[c].rjust(wd[c]) for c in range(len(HEAD))))


def _colors(k):
    import matplotlib
    return matplotlib.colormaps["viridis"](np.linspace(0, 0.9, max(k, 1)))


def draw_current(ax, results, p):
    """График тока через ключ S на заданной оси matplotlib."""
    xmax = 0
    for c, (n, t, i, v, m) in zip(_colors(len(results)), results):
        ax.plot(t * 1e6, i, color=c, lw=1.6, label=f"{n} секц.")
        ax.plot([m["t1"] * 1e6, m["t2"] * 1e6], [m["half"]] * 2, "o", color=c, ms=5)
        ax.plot(m["tA"] * 1e6, m["A"], "s", color=c, ms=5)
        xmax = max(xmax, m["tend"] * 1e6)
    ax.set_xlim(0, xmax * 1.1)
    ax.set_xlabel("t, мкс"); ax.set_ylabel("i_S, А")
    ax.set_title(f"Ток через ключ S при замыкании  (Um = {p['um']:g} В, R1 = {p['r1']:g} Ом, XL1 = {p['xl1']:g} Ом, "
                 f"R2 = {p['r2']:g} Ом, XL2 = {p['xl2']:g} Ом, C = {p['c']*1e6:g} мкФ)", fontsize=9)
    ax.grid(alpha=.3); ax.legend(title="■ — Imax, ● — уровень A/2", fontsize=8, ncol=2)


def draw_voltage(ax, results, p=None):
    """Напряжения на конденсаторах первой и последней секций."""
    xmax = 0
    for c, (n, t, i, v, m) in zip(_colors(len(results)), results):
        ax.plot(t * 1e6, v[:, 0], color=c, lw=1.3, label=f"{n} секц.: C1")
        if n > 1:
            ax.plot(t * 1e6, v[:, -1], color=c, lw=1.0, ls="--", label=f"{n} секц.: C{n}")
        xmax = max(xmax, m["tend"] * 1e6)
    ax.set_xlim(0, xmax * 1.1); ax.set_xlabel("t, мкс"); ax.set_ylabel("u_C, В")
    ax.set_title("Напряжение на конденсаторах первой и последней секций", fontsize=10)
    ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)


def make_figure(results, p, path, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4.8 + 0.34 * (len(results) + 2)))
    gs = fig.add_gridspec(2, 1, height_ratios=[4.8, 0.34 * (len(results) + 2)])
    draw_current(fig.add_subplot(gs[0]), results, p)
    axt = fig.add_subplot(gs[1]); axt.axis("off")
    tb = axt.table(cellText=[row(m) for *_, m in results], colLabels=HEAD, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.35)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    if show:
        plt.show()


def make_voltage_figure(results, p, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    draw_voltage(ax, results, p)
    fig.tight_layout(); fig.savefig(path, dpi=120)


# --------------------------------------------------------------------------- #
#                               ВВОД ПАРАМЕТРОВ                               #
# --------------------------------------------------------------------------- #
def ask(prompt, default, conv=float):
    s = input(f"{prompt} [{default}]: ").strip().replace(",", ".")
    return conv(s) if s else default


def interactive():
    print("Введите параметры (Enter - значение по умолчанию).")
    d = {}
    d["um"] = ask("Амплитуда напряжения Um, В", 220.0)
    d["r1"] = ask("R секции 1, Ом", 0.6)
    d["xl1"] = ask("XL секции 1, Ом", 0.2)
    d["r2"] = ask("R секций 2...n, Ом", 0.01)
    d["xl2"] = ask("XL секций 2...n, Ом", 0.003)
    d["c"] = ask("C каждой секции, мкФ", 10.0)
    d["w"] = ask("Угловая частота w, рад/с", 314.0)
    d["phase"] = ask("Начальная фаза, град", 90.0)
    s = input("Числа секций через пробел [1 2 3 5 10 20 30 50]: ").strip().replace(",", " ")
    d["sections"] = [int(x) for x in s.split()] if s else [1, 2, 3, 5, 10, 20, 30, 50]
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--um", type=float, default=220.0, help="амплитуда напряжения, В")
    ap.add_argument("--r1", type=float, default=0.6, help="R секции 1, Ом")
    ap.add_argument("--xl1", type=float, default=0.2, help="XL секции 1, Ом")
    ap.add_argument("--r2", type=float, default=0.01, help="R секций 2...n, Ом")
    ap.add_argument("--xl2", type=float, default=0.003, help="XL секций 2...n, Ом")
    ap.add_argument("--c", type=float, default=10.0, help="C каждой секции, мкФ")
    ap.add_argument("--w", type=float, default=314.0, help="угловая частота, рад/с")
    ap.add_argument("--phase", type=float, default=90.0, help="начальная фаза, град")
    ap.add_argument("--sections", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 30, 50])
    ap.add_argument("--tmax", type=float, default=60.0, help="макс. время расчёта, мс")
    ap.add_argument("--out", default="lrc_result", help="префикс выходных файлов")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--show", action="store_true", help="показать окно с графиком")
    if len(sys.argv) == 1:
        d = interactive()
        args = argparse.Namespace(**d, tmax=60.0, out="lrc_result", workers=os.cpu_count() or 1, show=True)
    else:
        args = ap.parse_args()

    p = dict(um=args.um, r1=args.r1, xl1=args.xl1, r2=args.r2, xl2=args.xl2,
             c=args.c * 1e-6, w=args.w, phase=args.phase, tmax=args.tmax * 1e-3)
    secs = sorted(set(args.sections))
    print(f"Расчёт для секций: {secs} ...", flush=True)
    jobs = [(n, p) for n in secs]
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(run_one, jobs))
    else:
        results = [run_one(j) for j in jobs]
    results.sort(key=lambda r: r[0])

    rows = [row(m) for *_, m in results]
    print_table(rows)
    with open(args.out + "_table.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f, delimiter=";"); wr.writerow(HEAD); wr.writerows(rows)
    make_figure(results, p, args.out + "_current.png", show=args.show)
    make_voltage_figure(results, p, args.out + "_voltage.png")
    print(f"\nСохранено: {args.out}_table.csv, {args.out}_current.png, {args.out}_voltage.png")


if __name__ == "__main__":
    main()
