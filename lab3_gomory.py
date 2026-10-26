#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота 3. Метод дробових відсічень Гоморі (повністю цілочисельне ЛП).

    max F = c^T x,   A x <= b,   x >= 0,   x - цілочисельний вектор,
    A, b, c - цілі (для базового класу b >= 0; від'ємні b також допускаються,
    тоді релаксація розв'язується двофазним методом).

Алгоритм
  1. Відкидаємо цілочисельність; релаксацію розв'язуємо симплекс-методом
     з ЛР1 (модуль simplex.py). Цілочисельний оптимум -> кінець.
  2. Обираємо рядок таблиці з найбільшою дробовою частиною правої частини
     (нічия -> менший номер рядка). ЗАСТОСОВНІСТЬ: усі змінні рядка (початкові,
     додаткові та змінні відсічень) мають бути цілочисельними - це перевіряється.
  3. Відсічення  sum_j {a_rj} x_j >= {b_r}  ({.} - дробова частина).
     Щоб нова додаткова змінна була цілочисельною за побудовою, відсічення
     множиться на спільний знаменник D коефіцієнтів:
         sum_j (D{a_rj}) x_j - t = D{b_r},  t >= 0,  t - ціле.
  4. Рядок додається до таблиці; план стає недопустимим (t = -D{b_r} < 0),
     але двоїстодопустимим -> повторна оптимізація ДВОЇСТИМ симплекс-методом.
  5. Повторюємо до цілочисельного плану; якщо двоїстий симплекс показує
     несумісність - цілочисельних допустимих планів немає. Ліміт відсічень
     вичерпано -> "незавершений пошук" (а не доведена відсутність розв'язку).

Усі обчислення - у раціональних числах (Fraction): похибок немає, eps = 0.
Просте округлення розв'язку релаксації НЕ використовується.

Запуск:
    python lab3_gomory.py                 # інтерактивне введення
    python lab3_gomory.py --demo          # контрольний приклад
    python lab3_gomory.py --json data.json   # {"c": [..], "A": [[..]], "b": [..]}
    python lab3_gomory.py --test          # тести + перебір + scipy.milp
    додатково: --quiet (без таблиць), --max-cuts N
"""
import argparse
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass, field
from fractions import Fraction as Fr
from typing import List, Optional

from simplex import (INFEASIBLE, ITER_LIMIT, OPTIMAL, UNBOUNDED, Tableau, fmt,
                     solve_lp, to_fr)


# ============================================================================
# введення та перевірка
# ============================================================================
class InputError(ValueError):
    pass


def validate(c, A, b):
    if not c or not A or not b:
        raise InputError("c, A, b не можуть бути порожніми")
    n, m = len(c), len(A)
    if len(b) != m or any(len(r) != n for r in A):
        raise InputError(f"A має бути {m}x{n}, b - довжини {m}")
    try:
        c = [to_fr(v) for v in c]
        A = [[to_fr(v) for v in r] for r in A]
        b = [to_fr(v) for v in b]
    except ValueError as e:
        raise InputError(str(e))
    for name, vec in (("c", c), ("A", [v for r in A for v in r]), ("b", b)):
        if any(v.denominator != 1 for v in vec):
            raise InputError(f"елементи {name} мають бути цілими (повністю цілочисельна задача)")
    return [int(v) for v in c], [[int(v) for v in r] for r in A], [int(v) for v in b]


def frac(q: Fr) -> Fr:
    """Дробова частина {q} = q - floor(q), завжди в [0, 1)."""
    return q - math.floor(q)


def lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


# ============================================================================
# вирази змінних через початкові x (для запису відсічення в x-змінних)
# ============================================================================
def lin_str(coefs, names, const=Fr(0)) -> str:
    parts = []
    for k, nm in zip(coefs, names):
        if k == 0:
            continue
        sign = "-" if k < 0 else "+"
        mag = abs(k)
        term = nm if mag == 1 else f"{fmt(mag)}·{nm}"
        parts.append((sign, term))
    if const != 0:
        parts.append(("-" if const < 0 else "+", fmt(abs(const))))
    if not parts:
        return "0"
    s = ("-" if parts[0][0] == "-" else "") + parts[0][1]
    for sign, term in parts[1:]:
        s += f" {sign} {term}"
    return s


@dataclass
class Cut:
    number: int
    row_var: str            # базисна змінна обраного рядка
    row_value: Fr
    fractions: dict         # {назва змінної: {a_rj}}
    f0: Fr
    D: int
    var: str                # нова змінна t_k
    in_x: str               # відсічення в початкових змінних
    relax_value_after: Fr


@dataclass
class GomoryResult:
    status: str                       # optimal | infeasible | unbounded | iteration_limit
    x: Optional[List[int]] = None
    fun: Optional[int] = None
    relax_x: Optional[List[Fr]] = None
    relax_fun: Optional[Fr] = None
    cuts: List[Cut] = field(default_factory=list)
    simplex_iterations: int = 0
    message: str = ""


# ============================================================================
# метод Гоморі
# ============================================================================
def gomory(c, A, b, max_cuts=200, verbose=True, out=print) -> GomoryResult:
    c, A, b = validate(c, A, b)
    n, m = len(c), len(A)
    log = out if verbose else (lambda *_: None)

    log("Задача: max F = " + lin_str(c, [f"x{j + 1}" for j in range(n)]))
    for i in range(m):
        log(f"   {lin_str(A[i], [f'x{j + 1}' for j in range(n)])} <= {b[i]}")
    log(f"   x >= 0, x - цілі\n")

    # --- крок 1: лінійна релаксація (симплекс ЛР1)
    log("=== КРОК 1. Лінійна релаксація (цілочисельність відкинуто) ===")
    lp = solve_lp(c, A, b, ["<="] * m, maximize=True, verbose=verbose, out=out)
    if lp.status == INFEASIBLE:
        msg = "Система обмежень несумісна вже для релаксації -> цілочисельних планів немає."
        log("\n" + msg)
        return GomoryResult(INFEASIBLE, message=msg)
    if lp.status == UNBOUNDED:
        msg = ("Релаксація необмежена. Метод Гоморі потребує обмеженої релаксації "
               "(для раціональних даних й допустимої цілочисельної задачі F теж необмежена).")
        log("\n" + msg)
        return GomoryResult(UNBOUNDED, message=msg)
    if lp.status != OPTIMAL:
        return GomoryResult(ITER_LIMIT, message="ліміт ітерацій симплекс-методу")

    T: Tableau = lp.tableau
    T.verbose, T.out = False, out
    relax_x, relax_fun = list(lp.x), lp.fun
    log(f"\nОптимум релаксації: x = ({', '.join(fmt(v) for v in relax_x)}), "
        f"F = {fmt(relax_fun)}" + (f" ≈ {float(relax_fun):.4f}" if relax_fun.denominator != 1 else ""))

    # вирази всіх змінних через x:  var = const + sum k_i x_i
    expr = {}
    for j in range(n):
        expr[f"x{j + 1}"] = (Fr(0), [Fr(1 if k == j else 0) for k in range(n)])
    for i in range(m):                   # s_i = b_i - A_i x
        expr[f"s{i + 1}"] = (Fr(b[i]), [Fr(-A[i][k]) for k in range(n)])
    xnames = [f"x{j + 1}" for j in range(n)]

    cuts: List[Cut] = []
    base_iters = T.iterations
    res = lambda st, **kw: GomoryResult(st, relax_x=relax_x, relax_fun=relax_fun, cuts=cuts,
                                        simplex_iterations=T.iterations, **kw)

    while True:
        # --- перевірка цілочисельності
        if all(v.denominator == 1 for v in T.b):
            vals = T.values()
            x = [int(vals[j]) for j in range(n)]
            F = sum(ci * xi for ci, xi in zip(c, x))
            assert Fr(F) == T.zv
            log(f"\nПлан цілочисельний -> оптимум знайдено. Відсічень: {len(cuts)}.")
            return res(OPTIMAL, x=x, fun=F)
        if len(cuts) >= max_cuts:
            msg = (f"Досягнуто ліміт відсічень ({max_cuts}): пошук НЕ завершено "
                   f"(це не доводить відсутності розв'язку).")
            log("\n" + msg)
            return res(ITER_LIMIT, message=msg)

        # --- вибір рядка: найбільша дробова частина (нічия -> менший номер)
        fr_parts = [(frac(T.b[i]), i) for i in range(len(T.b))]
        f0, r = max(fr_parts, key=lambda p: (p[0], -p[1]))
        k = len(cuts) + 1
        log(f"\n=== ВІДСІЧЕННЯ №{k} ===")
        frac_rows = ", ".join(f"{T.names[T.basis[i]]}={fmt(T.b[i])}" for f, i in fr_parts if f > 0)
        log(f"Дробові базисні змінні: {frac_rows}")
        log(f"Обрано рядок {r + 1}: {T.names[T.basis[r]]} = {fmt(T.b[r])}, {{b}} = {fmt(f0)}")

        # --- обґрунтування застосовності
        used = [j for j in range(len(T.names)) if T.A[r][j] != 0]
        assert all(T.integer[j] for j in used), "рядок містить нецілу змінну - відсічення незастосовне"
        log("Застосовність: усі змінні рядка цілочисельні ("
            + ", ".join(T.names[j] for j in used) + ").")

        # --- побудова відсічення
        fj = [frac(T.A[r][j]) for j in range(len(T.names))]
        D = 1
        for q in fj + [f0]:
            D = lcm(D, q.denominator)
        coef = [D * q for q in fj]          # цілі
        rhs = D * f0                         # ціле
        assert all(q.denominator == 1 for q in coef) and rhs.denominator == 1
        tname = f"t{k}"
        nz = {T.names[j]: fj[j] for j in range(len(T.names)) if fj[j] != 0}
        log("Дробові частини коефіцієнтів: "
            + ", ".join(f"{{{nm}}}={fmt(v)}" for nm, v in nz.items()))
        log(f"Відсічення: " + " + ".join(f"{fmt(v)}·{nm}" for nm, v in nz.items())
            + f" >= {fmt(f0)}")
        log(f"Спільний знаменник D = {D}; ціла форма: "
            + " + ".join(f"{fmt(coef[j])}·{T.names[j]}" for j in range(len(T.names)) if coef[j] != 0)
            + f" - {tname} = {fmt(rhs)},  {tname} >= 0, {tname} - ціле")

        # --- запис у початкових змінних: t = sum coef_j * var_j - rhs
        const = -rhs
        kx = [Fr(0)] * n
        for j in range(len(T.names)):
            if coef[j] != 0:
                cj, kj = expr[T.names[j]]
                const += coef[j] * cj
                kx = [p + coef[j] * q for p, q in zip(kx, kj)]
        expr[tname] = (const, kx)
        g = 0
        for q in kx:
            g = math.gcd(g, int(q))
        in_x = f"{lin_str(kx, xnames)} >= {fmt(-const)}"
        if g > 1:
            in_x += f"   (ділимо на {g}: {lin_str([q / g for q in kx], xnames)} >= {fmt(-const / g)})"
        log("Відсічення в початкових змінних: " + in_x)

        # --- додати рядок та повторно оптимізувати двоїстим симплексом
        row_var, row_val = T.names[T.basis[r]], T.b[r]
        T.add_row(tname, [-q for q in coef], -rhs, integer=True)
        if verbose:
            T.show("Таблиця після додавання відсічення (план недопустимий: b < 0):")
        T.verbose = verbose
        st = T.dual()
        T.verbose = False
        cuts.append(Cut(k, row_var, row_val, nz, f0, D, tname, in_x, T.zv))
        if st == INFEASIBLE:
            msg = ("Двоїстий симплекс-метод: рядок з b<0 без від'ємних коефіцієнтів -> "
                   "цілочисельних допустимих планів немає.")
            log("\n" + msg)
            return res(INFEASIBLE, message=msg)
        if st != OPTIMAL:
            return res(ITER_LIMIT, message="ліміт ітерацій двоїстого симплексу")
        vals = T.values()
        log(f"Після реоптимізації: x = ({', '.join(fmt(vals[j]) for j in range(n))}), "
            f"F = {fmt(T.zv)}")


def report(res: GomoryResult, c, out=print):
    out("\n" + "=" * 60)
    names = {OPTIMAL: "ЦІЛОЧИСЕЛЬНИЙ ОПТИМУМ ЗНАЙДЕНО", INFEASIBLE: "ЦІЛОЧИСЕЛЬНИХ ПЛАНІВ НЕМАЄ",
             UNBOUNDED: "РЕЛАКСАЦІЯ НЕОБМЕЖЕНА", ITER_LIMIT: "НЕЗАВЕРШЕНИЙ ПОШУК"}
    out(f"Статус: {names[res.status]}")
    if res.relax_fun is not None:
        out(f"Релаксація: x = ({', '.join(fmt(v) for v in res.relax_x)}), F = {fmt(res.relax_fun)}")
    if res.status == OPTIMAL:
        out(f"Цілочисельний оптимум: x = {tuple(res.x)}, F = {res.fun}")
        gap = res.relax_fun - res.fun
        out(f"Різниця релаксація − ціла задача = {fmt(gap)}")
    out(f"Кількість відсічень: {len(res.cuts)}; ітерацій симплексу (разом): {res.simplex_iterations}")
    if res.cuts:
        out("Послідовність відсічень (у початкових змінних):")
        for ct in res.cuts:
            out(f"  №{ct.number}: {ct.in_x}")
    if res.message and res.status != OPTIMAL:
        out(res.message)


# ============================================================================
# незалежна перевірка: перебір і scipy.milp
# ============================================================================
def variable_bounds(c, A, b) -> Optional[List[int]]:
    """Верхні межі x_j із релаксації (max x_j); None, якщо хоч одна необмежена."""
    n, m = len(c), len(A)
    ub = []
    for j in range(n):
        r = solve_lp([1 if k == j else 0 for k in range(n)], A, b, ["<="] * m)
        if r.status == UNBOUNDED:
            return None
        if r.status == INFEASIBLE:
            return [0] * n
        ub.append(math.floor(r.fun))
    return ub


def brute_force(c, A, b, ub=None, limit=3_000_000):
    """Повний перебір у скінченних межах. Повертає (F, x) або None, якщо планів немає."""
    ub = ub or variable_bounds(c, A, b)
    if ub is None:
        raise ValueError("змінні необмежені - вкажіть межі перебору")
    if math.prod(u + 1 for u in ub) > limit:
        raise ValueError("простір перебору надто великий")
    best = None
    for x in itertools.product(*[range(u + 1) for u in ub]):
        if all(sum(a * xi for a, xi in zip(row, x)) <= bi for row, bi in zip(A, b)):
            F = sum(ci * xi for ci, xi in zip(c, x))
            if best is None or F > best[0]:
                best = (F, list(x))
    return best


def milp_check(c, A, b):
    """Незалежна перевірка scipy.optimize.milp (лише для контролю). -> (status, F)."""
    try:
        import numpy as np
        from scipy.optimize import LinearConstraint, Bounds, milp
    except ImportError:
        return None
    n = len(c)
    r = milp(c=-np.array(c, float), integrality=np.ones(n),
             constraints=LinearConstraint(np.array(A, float), -np.inf, np.array(b, float)),
             bounds=Bounds(0, np.inf))
    if r.status == 0:
        return "optimal", round(-r.fun)
    return {2: "infeasible", 3: "unbounded"}.get(r.status, "other"), None


def feasible(c, A, b, x) -> bool:
    return all(v >= 0 and v == int(v) for v in x) and \
        all(sum(a * xi for a, xi in zip(row, x)) <= bi for row, bi in zip(A, b))


# ============================================================================
# тести
# ============================================================================
def run_tests() -> bool:
    ok_all = True

    def case(title, c, A, b, expect=None, expect_status=OPTIMAL, expect_cuts=None):
        nonlocal ok_all
        res = gomory(c, A, b, verbose=False)
        errs = []
        if res.status != expect_status:
            errs.append(f"статус {res.status}, очікувався {expect_status}")
        bf = brute_force(c, A, b) if res.status != UNBOUNDED else None
        ml = milp_check(c, A, b)
        if res.status == OPTIMAL:
            if not feasible(c, A, b, res.x):
                errs.append("план порушує обмеження")
            if bf is None or bf[0] != res.fun:
                errs.append(f"перебір: {bf}")
            if ml and (ml[0] != "optimal" or ml[1] != res.fun):
                errs.append(f"milp: {ml}")
            if res.fun > res.relax_fun:
                errs.append("F цілої задачі більша за оптимум релаксації")
        elif res.status == INFEASIBLE:
            if bf is not None:
                errs.append(f"перебір знайшов план {bf}")
            if ml and ml[0] != "infeasible":
                errs.append(f"milp: {ml}")
        if expect is not None and res.fun != expect:
            errs.append(f"очікувано F={expect}, фактично {res.fun}")
        if expect_cuts is not None and len(res.cuts) != expect_cuts:
            errs.append(f"очікувано відсічень {expect_cuts}, фактично {len(res.cuts)}")
        ok = not errs
        ok_all &= ok
        rel = fmt(res.relax_fun) if res.relax_fun is not None else "-"
        print(f"[{'OK ' if ok else 'ERR'}] {title:<44} статус={res.status:<10} "
              f"релакс.F={rel:>6}  F={res.fun!s:>4}  x={res.x}  відсічень={len(res.cuts)}"
              + ("" if ok else "\n      " + "; ".join(errs)))

    print("=== Детерміновані тести ===")
    case("Контрольний приклад", [1, 1], [[2, 1], [1, 2]], [4, 4], expect=2)
    case("Одразу цілочисельна релаксація", [3, 2], [[1, 1], [2, 1]], [4, 5], expect=9, expect_cuts=0)
    case("Класичний (Wolsey): релакс. 63", [7, 9], [[-1, 3], [7, 1]], [6, 35], expect=58)
    case("Задача про рюкзак (одне обмеження)", [8, 11, 6, 4], [[5, 7, 4, 3]], [14], expect=21)
    case("Без цілих планів: 2x1+2x2 = 1", [1, 1], [[2, 2], [-2, -2]], [1, -1],
         expect_status=INFEASIBLE)
    case("Без цілих планів: 3x1+3x2 в [1;2]", [1, 2], [[3, 3], [-3, -3]], [2, -1],
         expect_status=INFEASIBLE)
    case("Несумісна вже релаксація", [1, 1], [[1, 1], [-1, -1]], [1, -3], expect_status=INFEASIBLE)
    case("Від'ємна права частина (допустимі цілі)", [1, 1], [[2, 2], [-1, 0]], [5, -1], expect=2)
    case("3 змінні", [5, 4, 3], [[2, 3, 1], [4, 1, 2], [3, 4, 2]], [5, 11, 8], expect=13)
    case("Необмежена релаксація", [1, 0], [[-1, 1]], [1], expect_status=UNBOUNDED)

    print("\n=== Тести коректності введення ===")
    bad = [("дробовий коефіцієнт", [1, 1], [[1.5, 1]], [3]),
           ("неправильні розміри", [1, 1], [[1, 1, 1]], [3]),
           ("нечислове значення", [1, "a"], [[1, 1]], [3]),
           ("порожні дані", [], [], [])]
    for title, c, A, b in bad:
        try:
            gomory(c, A, b, verbose=False)
            ok = False
        except InputError:
            ok = True
        ok_all &= ok
        print(f"[{'OK ' if ok else 'ERR'}] відхилено: {title}")

    print("\n=== Випадкові тести (500 задач): Гоморі vs повний перебір vs scipy.milp ===")
    rnd = random.Random(2024)
    bad_cnt = tot_cuts = max_cuts_seen = n_opt = n_inf = n_lim = 0
    for t in range(500):
        n, m = rnd.randint(2, 3), rnd.randint(1, 3)
        A = [[rnd.randint(-2, 6) for _ in range(n)] for _ in range(m)]
        b = [rnd.randint(0, 25) for _ in range(m)]
        if rnd.random() < 0.15:
            b[rnd.randrange(m)] = rnd.randint(-6, -1)
        # для обмеженості додаємо верхню межу суми
        A.append([1] * n)
        b.append(rnd.randint(3, 12))
        c = [rnd.randint(0, 9) for _ in range(n)]
        res = gomory(c, A, b, max_cuts=300, verbose=False)
        errs = []
        bf = brute_force(c, A, b)
        if res.status == OPTIMAL:
            n_opt += 1
            if not feasible(c, A, b, res.x) or bf is None or bf[0] != res.fun:
                errs.append(f"F={res.fun}, перебір={bf}")
            tot_cuts += len(res.cuts)
            max_cuts_seen = max(max_cuts_seen, len(res.cuts))
        elif res.status == INFEASIBLE:
            n_inf += 1
            if bf is not None:
                errs.append(f"оголошено несумісною, перебір {bf}")
        elif res.status == ITER_LIMIT:
            n_lim += 1
            errs.append("ліміт відсічень")
        else:
            errs.append(f"статус {res.status}")
        ml = milp_check(c, A, b)
        if ml and not errs:
            if (res.status == OPTIMAL) != (ml[0] == "optimal") or \
                    (res.status == OPTIMAL and ml[1] != res.fun):
                errs.append(f"milp: {ml}")
        if errs:
            bad_cnt += 1
            print(f"  [ERR] #{t}: {errs}  c={c} A={A} b={b}")
    ok = bad_cnt == 0
    ok_all &= ok
    print(f"[{'OK ' if ok else 'ERR'}] помилок: {bad_cnt}; оптимум: {n_opt}, несумісних: {n_inf}, "
          f"ліміт: {n_lim}; сер. відсічень: {tot_cuts / max(n_opt, 1):.2f}; макс.: {max_cuts_seen}")
    print("\nПІДСУМОК:", "УСІ ТЕСТИ ПРОЙДЕНО" if ok_all else "Є ПОМИЛКИ")
    return ok_all


# ============================================================================
# інтерфейс
# ============================================================================
def read_numbers(prompt, count):
    while True:
        parts = input(prompt).replace(";", " ").replace(",", " ").split()
        if len(parts) == count:
            return parts
        print(f"  Потрібно рівно {count} чисел, отримано {len(parts)}. Спробуйте ще раз.")


def interactive():
    print("Метод Гоморі: max c·x, A x <= b, x >= 0, x - цілі (c, A, b - цілі числа).")
    while True:
        try:
            n = int(input("Кількість змінних n: "))
            m = int(input("Кількість обмежень m: "))
            if n < 1 or m < 1:
                raise ValueError
            break
        except ValueError:
            print("  Введіть натуральні числа.")
    while True:
        try:
            c = read_numbers(f"Коефіцієнти цільової функції c ({n} чисел): ", n)
            print(f"Матриця A ({m} рядків по {n} чисел):")
            A = [read_numbers(f"  A[{i + 1}]: ", n) for i in range(m)]
            b = read_numbers(f"Праві частини b ({m} чисел): ", m)
            validate(c, A, b)
            return c, A, b
        except InputError as e:
            print(f"  Помилка даних: {e}. Введіть дані ще раз.")


def main():
    ap = argparse.ArgumentParser(description="ЛР3: метод Гоморі")
    ap.add_argument("--demo", action="store_true", help="контрольний приклад")
    ap.add_argument("--test", action="store_true", help="запустити тести")
    ap.add_argument("--json", help="файл JSON з полями c, A, b")
    ap.add_argument("--quiet", action="store_true", help="лише підсумок")
    ap.add_argument("--max-cuts", type=int, default=200)
    args = ap.parse_args()

    if args.test:
        sys.exit(0 if run_tests() else 1)
    if args.demo:
        c, A, b = [1, 1], [[2, 1], [1, 2]], [4, 4]
    elif args.json:
        with open(args.json, encoding="utf-8") as f:
            d = json.load(f)
        c, A, b = d["c"], d["A"], d["b"]
    else:
        c, A, b = interactive()
    try:
        res = gomory(c, A, b, max_cuts=args.max_cuts, verbose=not args.quiet)
    except InputError as e:
        sys.exit(f"Помилка даних: {e}")
    report(res, c)

    # незалежна перевірка
    cc, AA, bb = validate(c, A, b)
    if res.status != UNBOUNDED:
        try:
            bf = brute_force(cc, AA, bb)
            print("Перевірка повним перебором:",
                  "цілих допустимих планів немає" if bf is None else f"F = {bf[0]}, x = {tuple(bf[1])}")
        except ValueError as e:
            print(f"Перебір не виконано: {e}")
    ml = milp_check(cc, AA, bb)
    if ml:
        print(f"Незалежна перевірка scipy.milp: {ml[0]}" + (f", F = {ml[1]}" if ml[1] is not None else ""))


if __name__ == "__main__":
    main()
