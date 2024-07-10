import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from random import normalvariate
import warnings

warnings.filterwarnings('ignore', 'The iteration is not making good progress')


# вычисление мат. ожидания (1 и 5 задания)
def calculate_median(sequence):
    n = len(sequence)
    median = sum(sequence) / n  # считаем среднее
    return median  # выводим ответ


# функция вычисления дисперсии (1 и 5 задания)
def calculate_dispersion(sequence):
    n = len(sequence)
    median = calculate_median(sequence)
    # заполняем список квадратами разности наших чисел и выборочного среднего
    disp_list = [(item - median) * (item - median) for item in sequence]

    dispersion = sum(disp_list) / (n - 1)  # считаем дисперсию
    return dispersion  # выводим дисперсию


# корреляционная функция (1 и 5 задания)
def calculate_correlation(sequence, m):
    m = int(m // 1)
    sumR = 0
    median = calculate_median(sequence)
    n = len(sequence)
    for j in range(n - m):
        # производим суммирование по формуле
        sumR += (sequence[j] - median) * (sequence[m + j] - median)
    return sumR / (n - m - 1)


# НКФ (1 и 5 задания)
def calculate_normalCorrelation(sequence, k):
    return calculate_correlation(sequence, k) / calculate_dispersion(sequence)


# Интервал корреляции (1 задание)
def calculate_correlation_interval(sequence):
    cf = 0.01  # вводим коэффициент, чтоб корректно посчитать интервал
    t = cf * len(sequence) - 1
    exp = math.exp(-1)
    while abs(calculate_normalCorrelation(sequence, t)) < exp:
        t -= 1
    return t


# изображаем фрагмент исходного СП (1 и 6 задания)
def plot1(start_list, median, dispersion, labelname, name):
    sko = math.sqrt(dispersion)  # вычисляем ско

    x = start_list[:140].copy()  # берём первые 140 значений для построения
    fig, ax = plt.subplots(num=name)

    plt.plot(x, label=labelname)  # добавляем координаты на график

    minY = min(x) - 10
    maxY = max(x) + 10

    ax.set(xlim=(0, 140), ylim=(minY, maxY))  # корректируем границы графика

    # подписываем оси
    plt.xlabel("Index number")
    plt.ylabel("Random sequence values")

    plt.axhline(y=median, color='red', label="Average")  # добавляем на график прямую с y = median
    # стандартное отклонение
    plt.axhline(y=median + sko, color='blue', label="Standard deviation")
    plt.axhline(y=median - sko, color='blue')

    plt.grid()  # добавляем сетку
    plt.legend()  # добавляем легенду
    plt.show()  # выводим график


# графическая оценка НКФ (1 задание)
def plot2(normalCorrelation_list, m, corInt):
    fig, ax = plt.subplots(num="Графическая оценка НКФ")

    plt.plot(normalCorrelation_list, color="r", label="Source")  # добавляем координаты на график

    if corInt > m:
        m = corInt + 5

    ax.set(xlim=(-0.03, m), ylim=(-1, 1))  # корректируем границы графика

    # подписываем оси
    plt.xlabel("Index number")
    plt.ylabel("Normalized correlation function")

    plt.axvline(x=corInt, color='green', label='Interval of correlation')

    # добавляем на график прямые с y = 1/e и -1/e
    plt.axhline(y=math.exp(-1), color='blue', label="1/e and -1/e")
    plt.axhline(y=-math.exp(-1), color='blue')

    plt.grid()  # добавляем сетку
    plt.legend()  # добавляем легенду
    plt.show()  # выводим график


# поиск коэффициентов альфа и бета для моделей АР (2 и 5 задания)
def findAlphaBetasAR(R: list, size: int):
    a_list = []
    b_list = []

    # в цикле для каждого М будем составлять систему уравнений
    for M in range(0, size + 1):
        vec_b = []
        matr_a = []
        for m in range(0, M + 1):
            vec_b.append([R[m]])
            matr_buf = []

            if m == 0:
                matr_buf.append(1)
            else:
                matr_buf.append(0)

            for n in range(1, M + 1):
                matr_buf.append(R[abs(m - n)])
            matr_a.append(matr_buf)

        matr_a = np.array(matr_a)
        vec_b = np.array(vec_b)

        res = np.linalg.solve(matr_a, vec_b)  # решаем систему линейных уравнений Ax = b
        res = res.ravel()

        a_list.append([[math.sqrt(res[0])]])  # добавляем в список значение параметра альфа
        print(f"Для M = {M} параметр альфа = {math.sqrt(res[0])}")

        b_temp = []
        for i in range(1, M + 1):  # добавялем в список значения параметра бета
            b_temp.append(res[i])
            print(f"Для M = {M} параметр бета = {res[i]}")
        b_list.append([b_temp])

        print()

    return a_list, b_list


# вычисление теоретической НКФ для моделей АРСС (2, 3, 4, 5 задания)
def theoretical_normalCorrelation(b_list, a_list, normalCorrelation_list, end, M_max, N_max, M_min, N_min):
    def check():  # функция-заглушка для поиска тНКФ для моделей АРСС
        fixed_list = []
        if M_min == 1 and N_min == 1:
            fixed_list.append([])

        return fixed_list

    theoretical_list = check()  # создаём список для значений теоретической НКФ

    # в цикле для каждого M и N будем считать теоретические НКФ
    for M in range(M_min, M_max + 1):
        buf_list = check()

        for N in range(N_min, N_max + 1):
            if not a_list or not a_list[M][N][0] is None:
                sub_buf_list = []  # список для значений теоретической НКФ для конкретных M и N

                for m in range(end + 1):
                    if m <= M + N:  # если k <= M + N, то теоретическая НКФ совпадает с выборочной
                        sub_buf_list.append(normalCorrelation_list[m])
                        print(f"Для M = {M}, N = {N}, m = {m} теоретическая НКФ = {normalCorrelation_list[m]}")

                    else:  # считаем по формуле теоретическую НКФ
                        theoretical = 0
                        for j in range(1, M + 1):
                            theoretical += b_list[M][N][j - 1] * sub_buf_list[m - j]
                        print(f"Для M = {M}, N = {N}, m = {m} теоретическая НКФ = {theoretical}")
                        sub_buf_list.append(theoretical)

                buf_list.append(sub_buf_list)
                print()
            else:
                print(f"Для M = {M} и N = {N} модели не существует\n")
                buf_list.append([None])

        theoretical_list.append(buf_list)

    return theoretical_list


# вычисление погрешности для каждой из моделей (2, 3, 4, 5 задания)
def findEps(theoretical_nkf_list, normalCorrelation_list, end, M_max, N_max, M_min, N_min):
    def check():
        fixed_list = []
        if M_min == 1 and N_min == 1:
            fixed_list.append([])

        return fixed_list

    eps_list = check()  # список для значений погрешностей
    for M in range(M_min, M_max + 1):  # для каждого порядка будем считать погрешность
        buf_list = check()
        for N in range(N_min, N_max + 1):
            if not theoretical_nkf_list[M][N][0] is None:
                eps = 0
                for m in range(1, end + 1):  # считаем погрешность по формуле
                    eps += (theoretical_nkf_list[M][N][m] - normalCorrelation_list[m]) ** 2
                print(f"Для M = {M} и N = {N} погрешность = {eps}")
                buf_list.append(eps)  # добавляем в список нашу погрешность
            else:
                print(f"Для M = {M} и N = {N} модели не существует")
                buf_list.append(None)

        eps_list.append(buf_list)
        print()

    return eps_list


# поиск лучшей модели (2, 3, 4 задания)
def best_model(eps_list, M_max, N_max, M_min, N_min, stability):
    def check(m, n):
        result = True
        if M_max > 0 and N_max > 0:
            if stability[m][n] is None or stability[m][n] == False:
                result = False
            if stability[m][n] == False:
                print(f"Для M = {m} и N = {n} модель не устойчива\n")
        return result

    def init_min():
        if M_min == 1 and N_min == 1:
            return eps_list[1][1]
        return eps_list[0][0]

    min_eps = init_min()
    min_MN = (0, 0)

    for M in range(M_min, M_max + 1):
        for N in range(N_min, N_max + 1):
            if not eps_list[M][N] is None and check(M, N):
                if min_eps is None or eps_list[M][N] < min_eps:
                    min_eps = eps_list[M][N]
                    min_MN = (M, N)
            else:
                print(f"Для M = {M} и N = {N} модели не существует\n")

    print(f"Лучшая модель при M = {min_MN[0]} и N = {min_MN[1]} с эпсилон = {min_eps}")
    return min_MN


# правила для решения системы уравнений СС(N) для каждого порядка N (3 задание)
def equationsMA(seq, R):
    a = []
    factor_list = []

    n = len(seq) - 1

    for elem in seq:
        a.append(elem)

    for m in range(len(seq)):
        factor = 0
        for i in range(n - m + 1):
            factor += a[i] * a[i + m]
        factor -= R[m]

        factor_list.append(factor)

    return factor_list


# нахождение параметра альфа для моделей СС (3 и 5 задания)
def findAlphasMA(R, N_max):
    ans_list = []  # список для ответа

    # в цикле будем решать системы уравнений для каждого порядка N
    for n in range(N_max + 1):
        zeros = [0.0] * (n + 1)
        zeros[0] = math.sqrt(R[0])
        x0 = np.array(zeros)

        res = fsolve(equationsMA, x0, args=R)
        norm = np.linalg.norm(equationsMA(res, R))

        # проверка модели на существование и выдача нужного ответа
        if norm < 1e-4:
            buf_list = []
            for i in range(n + 1):
                buf_list.append(res[i])
                print(f"Для N = {n} параметр альфа = {res[i]}")
            ans_list.append(buf_list)
        else:
            print(f"Для N = {n} модели не существует")
            ans_list.append([None])
        print()

    return [ans_list]


# вспомогательная функция для вычисления коэффициентов альфа и бета моделей АРСС
# (4 задание)
def ARMA_factor_generator(sequence, R, M, N):
    a = []
    b = []
    Rxi = []

    # распределяем элементы по спискам
    for i in range(len(sequence)):
        cur_elem = sequence[i]
        if i < (N + 1):
            a.append(cur_elem)
        elif N < i < (N + M + 1):
            b.append(cur_elem)
        else:
            Rxi.append(cur_elem)

    factor_list = []  # список для правил
    # генерируем правила для системы типа Таблица 1.1 А
    for n in range(N + 1):
        factor = 0
        for j in range(1, M + 1):
            factor += b[j - 1] * R[abs(n - j)]
        for i in range(n, N + 1):
            factor += a[i] * Rxi[i - n]
        factor -= R[n]

        factor_list.append(factor)

    # генерируем правила для системы типа Таблица 1.1 Б
    for i in range(1, M + 1):
        factor = 0
        for j in range(1, M + 1):
            factor += b[j - 1] * R[abs(N - j + i)]
        factor -= R[N + i]

        factor_list.append(factor)

    # генерируем правила для системы типа Таблица 1.1 Г
    for n in range(N + 1):
        factor = 0
        m = min(n, M)
        if n > 0:
            for j in range(1, m + 1):
                factor += b[j - 1] * Rxi[n - j]
        factor += (a[n] - Rxi[n])

        factor_list.append(factor)

    return factor_list


# функция для вычисления коэффициентов альфа и бета моделей АРСС (4 и 5 задания)
def findBetasAlphasARMA(R, M_max, N_max):
    a_list = [[]]
    b_list = [[]]
    for m in range(1, M_max + 1):
        buf_a_list = [[]]
        buf_b_list = [[]]
        for n in range(1, N_max + 1):
            sub_a_list = []
            sub_b_list = []

            p = [0] * (m + 2 * n + 2)
            p[0] = math.sqrt(R[0])
            x0 = np.array(p)

            res = fsolve(ARMA_factor_generator, x0, args=(R, m, n))
            norm = np.linalg.norm(ARMA_factor_generator(res, R, m, n))
            tern = ARMA_factor_generator(res, R, m, n)

            if norm < 1e-4:
                for index in range(n + 1):
                    elem = res[index]

                    print(f"Для M = {m} и N = {n} коэффициент альфа = {elem} норма = {norm}")
                    sub_a_list.append(elem)
                print()
                for indx in range(m):
                    elem = res[indx + n + 1]

                    print(f"Для M = {m} и N = {n} коэффициент бета = {elem}")
                    sub_b_list.append(elem)
                print()
            else:
                print(f"Для M = {m} и N = {n} модели не существует\n")
                sub_a_list.append(None)
                sub_b_list.append(None)

            buf_a_list.append(sub_a_list)
            buf_b_list.append(sub_b_list)

        a_list.append(buf_a_list)
        b_list.append(buf_b_list)

    return a_list, b_list


# функция для проверки моделей на стабильность (4 задание)
def checkStability(b_list, M_max, N_max, M_min, N_min):
    result = [[]]
    for M in range(M_min, M_max + 1):
        sub_result = [[]]
        for N in range(N_min, N_max + 1):
            if not b_list[M][N][0] is None:
                betas = b_list[M][N]
                size = len(betas)

                answer = False
                if size == 0:
                    answer = True
                elif size == 1:
                    answer = abs(betas[0]) < 1
                elif size == 2:
                    answer = abs(betas[1]) < 1 and abs(betas[0]) < 1 - betas[1]
                elif size == 3:
                    statement1 = abs(betas[2]) < 1
                    statement2 = abs(betas[0] + betas[2]) < 1 - betas[1]
                    statement3 = abs(betas[1] + betas[0] * betas[2]) < 1 - betas[2] ** 2

                    answer = statement1 and statement2 and statement3
                if answer:
                    print(f"Для M = {M} и N = {N} модель устойчива")
                else:
                    print(f"Для M = {M} и N = {N} модель не устойчива")

                sub_result.append(answer)

            else:
                print(f"Для M = {M} и N = {N} модели не существует")
                sub_result.append(None)

        print()

        result.append(sub_result)

    return result


# функция для генерации выборки из old_n значений (5 задание)
def generate_seq(alphas_list, betas_list, M, N, old_n, median):
    def check():
        if not betas_list:
            return []
        return betas_list[M][N]

    a_list = alphas_list[M][N]
    b_list = check()

    badCount = 1000
    new_n = old_n + badCount

    eta_list = [0] * new_n
    ksi_list = [normalvariate(0, 1) for _ in range(new_n)]

    for n in range(new_n):
        for j in range(1, M + 1):
            if n - j >= 0:
                eta_list[n] += b_list[j - 1] * eta_list[n - j]

        for i in range(N + 1):
            if n - i >= 0:
                eta_list[n] += a_list[i] * ksi_list[n - i]

    eta_list = eta_list[1000:]

    for n in range(old_n):
        eta_list[n] += median

    return eta_list


# дополнительное слагаемое для значений генерируемой последовательности (5 задание)
def getExtraME(betas, median, m, n):
    if not betas:
        return median
    return median * (1 - sum(betas[m][n]))


# графическое сравнение НКФ смоделированного и исходного СП (5 задание)
def plot3(nkf_source, nkf_model, tNKF_model, m, name):
    fig, ax = plt.subplots(num=f"Графическая оценка НКФ модели {name}")

    plt.plot(nkf_model, color="g", label="Modeling")  # добавляем НКФ нового СП
    plt.plot(nkf_source, color="r", label="Source")  # добавляем НКФ исходного СП
    plt.plot(tNKF_model, color="b", label="Theoretical")  # добавляем тНКФ нового СП

    ax.set(xlim=(0, m), ylim=(-1, 1))  # корректируем границы графика

    # подписываем оси
    plt.xlabel("Index number")
    plt.ylabel("Normalized correlation function")

    plt.grid()  # добавляем сетку
    plt.legend()  # добавляем легенду
    plt.show()  # выводим график


# выбор лучшей модели из моделей АР, СС и АРСС (6 задание)
def best_of_the_best(eps_th_list, eps_list, models):
    min_th_eps = min(eps_th_list)

    min_mod_eps = min(eps_list)

    min_eps = min(min_th_eps, min_mod_eps)
    if min_eps in eps_th_list:
        min_index = eps_th_list.index(min_eps)
    else:
        min_index = eps_list.index(min_eps)

    best_m = models[min_index][0]
    best_n = models[min_index][1]

    print(f"Лучшая модель при M = {best_m} и N = {best_n} с эпсилон = {min_eps}")

    return min_eps, best_m, best_n, min_index


# главная функция, откуда всё вызывается
def main():
    f = open('16.txt', 'r')  # открываем файл, из которого будем считывать данные
    start_list = [float(line) for line in f]  # заполняем список числами из файла
    length = len(start_list)
    m = 10  # задаём количество отсчётов как константу
    M_max = 3  # максимальный порядок M
    N_max = 3  # максимальный порядок N
    M_min = 0  # минимальный порядок M
    N_min = 0  # минимальный порядок N

    print("\n-------------------------Задание №1-------------------------\n")

    median = calculate_median(start_list)  # получаем значение медианы
    print(f"Медиана = {median}\n")  # выводим её значение на экран

    dispersion = calculate_dispersion(start_list)  # получаем значение дисперсии
    print(f"Дисперсия = {dispersion}\n")  # выводим её значение на экран

    correlation_interval = calculate_correlation_interval(start_list)
    print(f"Интервал корреляции = {correlation_interval}\n")  # выводим на экран значение интервала

    normalCorrelation_list = []  # заводим список для НКФ
    correlation_list = []  # заводим список для КФ

    print("----------Значения корреляционной функции и НКФ----------\n")
    # в цикле вычисляем значения корреляционной функции и НКФ
    for i in range(0, m + 1):
        correlation = calculate_correlation(start_list, i)
        correlation_list.append(correlation)
        print(f"Для m = {i} корреляционная функция = {correlation}")  # выводим значение корреляции на экран

        normalCorrelation = calculate_normalCorrelation(start_list, i)
        normalCorrelation_list.append(normalCorrelation)
        print(f"Для m = {i} НКФ = {normalCorrelation}\n")  # выводим значение НКФ на экран

    plot1(start_list, median, dispersion, "Source process", "Фрагмент исходного СП")  # изображаем фрагмент исходного СП
    plot2(normalCorrelation_list, m, correlation_interval)  # графическая оценка НКФ

    print("-------------------------Задание №2-------------------------\n")

    # получаем значения параметров альфа и бета
    print("------------Значения параметров альфа и бета------------\n")
    a_listAR, b_listAR = findAlphaBetasAR(correlation_list, M_max)

    # вычисляем теоретическую НКФ для АР
    print("-----------------Теоретическая НКФ для АР-----------------\n")
    theoretical_nkf_listAR = theoretical_normalCorrelation(b_listAR, [], normalCorrelation_list, m, M_max, N_min, M_min,
                                                           N_min)

    # находим эпсилон для каждой модели
    print("----------------Эпсилон для каждой модели----------------\n")
    eps_listAR = findEps(theoretical_nkf_listAR, normalCorrelation_list, m, M_max, N_min, M_min, N_min)

    # выбор лучшей модели АР
    print("--------------------Выбор лучшей модели--------------------\n")
    ar_m, ar_n = best_model(eps_listAR, M_max, N_min, M_min, N_min, None)

    print("-------------------------Задание №3-------------------------\n")

    # получаем значения параметра альфа
    print("-----------------Значения параметра альфа-----------------\n")
    a_listMA = findAlphasMA(correlation_list, 3)

    # вычисляем теоретическую НКФ для СС
    print("-----------------Теоретическая НКФ для СС-----------------\n")
    theoretical_nkf_listMA = theoretical_normalCorrelation([], a_listMA, normalCorrelation_list, m, M_min, N_max, M_min,
                                                           N_min)

    # находим эпсилон для каждой модели
    print("----------------Эпсилон для каждой модели----------------\n")
    eps_listMA = findEps(theoretical_nkf_listMA, normalCorrelation_list, m, M_min, N_max, M_min, N_min)

    # выбор лучшей модели СС
    print("--------------------Выбор лучшей модели--------------------\n")
    ma_m, ma_n = best_model(eps_listMA, M_min, N_max, M_min, N_min, None)

    print("-------------------------Задание №4-------------------------\n")

    # находим коэффициенты бета и альфа для каждой модели АРСС
    print("------------Значения параметров альфа и бета------------\n")
    a_listARMA, b_listARMA = findBetasAlphasARMA(correlation_list, M_max, N_max)

    # вычисляем теоретическую НКФ для АРСС
    print("----------------Теоретическая НКФ для АРСС----------------\n")
    theoretical_nkf_listARMA = theoretical_normalCorrelation(b_listARMA, a_listARMA, normalCorrelation_list, m, M_max,
                                                             N_max, 1, 1)

    # находим эпсилон для каждой модели
    print("----------------Эпсилон для каждой модели----------------\n")
    eps_listARMA = findEps(theoretical_nkf_listARMA, normalCorrelation_list, m, M_max, N_max, 1, 1)

    # проверяем все модели на стабильность
    print("-------------Проверка моделей на стабильность-------------\n")
    stability_list = checkStability(b_listARMA, M_max, N_max, 1, 1)

    # выбор лучшей модели АРСС
    print("--------------------Выбор лучшей модели--------------------\n")
    arma_m, arma_n = best_model(eps_listARMA, M_max, N_max, 1, 1, stability_list)

    print("-------------------------Задание №5-------------------------\n")

    # генерируем по 5000 значений для каждой модели
    ar = generate_seq(a_listAR, b_listAR, ar_m, ar_n, length, median)
    ma = generate_seq(a_listMA, [], ma_m, ma_n, length, median)
    arma = generate_seq(a_listARMA, b_listARMA, arma_m, arma_n, length, median)

    print("----------------Значения моментных функций----------------\n")
    # матожидание, дисперсия, ско, НКФ, тНКФ
    medianAR = calculate_median(ar)
    medianMA = calculate_median(ma)
    medianARMA = calculate_median(arma)

    print(f"Для АР({ar_m}) медиана = {medianAR}")  # выводим её значение на экран
    print(f"Для СС({ma_n}) медиана = {medianMA}")  # выводим её значение на экран
    print(f"Для АРСС({arma_m}, {arma_n}) медиана = {medianARMA}\n")  # выводим её значение на экран

    dispAR = calculate_dispersion(ar)
    dispMA = calculate_dispersion(ma)
    dispARMA = calculate_dispersion(arma)

    print(f"Для АР({ar_m}) дисперсия = {dispAR}")  # выводим её значение на экран
    print(f"Для СС({ma_n}) дисперсия = {dispMA}")  # выводим её значение на экран
    print(f"Для АРСС({arma_m}, {arma_n}) дисперсия = {dispARMA}\n")  # выводим её значение на экран

    sko = math.sqrt(dispersion)
    skoAR = math.sqrt(dispAR)
    skoMA = math.sqrt(dispMA)
    skoARMA = math.sqrt(dispARMA)

    print(f"Для исходного СП СКО = {sko}")  # выводим её значение на экран
    print(f"Для АР({ar_m}) СКО = {skoAR}")  # выводим её значение на экран
    print(f"Для СС({ma_n}) СКО = {skoMA}")  # выводим её значение на экран
    print(f"Для АРСС({arma_m}, {arma_n}) СКО = {skoARMA}\n")  # выводим её значение на экран

    correlation_listAR = []
    correlation_listMA = []
    correlation_listARMA = []

    normalCorrelation_listAR = []
    normalCorrelation_listMA = []
    normalCorrelation_listARMA = []

    print("----------Значения корреляционной функции и НКФ----------\n")

    # в цикле вычисляем значения корреляционной функции и НКФ
    for i in range(0, m + 1):
        correlationAR = calculate_correlation(ar, i)
        correlation_listAR.append(correlationAR)
        print(f"Для m = {i} корреляционная функция АР = {correlationAR}")  # выводим значение корреляции на экран

        normalCorrelationAR = calculate_normalCorrelation(ar, i)
        normalCorrelation_listAR.append(normalCorrelationAR)
        print(f"Для m = {i} НКФ АР = {normalCorrelationAR}\n")  # выводим значение НКФ на экран

        correlationMA = calculate_correlation(ma, i)
        correlation_listMA.append(correlationMA)
        print(f"Для m = {i} корреляционная функция СС = {correlationMA}")  # выводим значение корреляции на экран

        normalCorrelationMA = calculate_normalCorrelation(ma, i)
        normalCorrelation_listMA.append(normalCorrelationMA)
        print(f"Для m = {i} НКФ СС = {normalCorrelationMA}\n")  # выводим значение НКФ на экран

        correlationARMA = calculate_correlation(arma, i)
        correlation_listARMA.append(correlationARMA)
        print(f"Для m = {i} корреляционная функция АР = {correlationARMA}")  # выводим значение корреляции на экран

        normalCorrelationARMA = calculate_normalCorrelation(arma, i)
        normalCorrelation_listARMA.append(normalCorrelationARMA)
        print(f"Для m = {i} НКФ АРСС = {normalCorrelationARMA}\n")  # выводим значение НКФ на экран

    eps_th_list = []  # отсюда будем выбирать лучшую модель
    eps_list = []

    print(f"--------------------Данные модели АР{ar_m, ar_n}--------------------\n")

    # получаем значения параметров альфа и бета
    print("------------Значения параметров альфа и бета------------\n")
    alphasAR, betasAR = findAlphaBetasAR(correlation_listAR, M_max)

    extraMEAR = getExtraME(betasAR, median, ar_m, ar_n)
    print(f"К значениям последовательности АР{ar_m, ar_n} необходимо добавить {extraMEAR}\n")

    # вычисляем теоретическую НКФ для АР
    print("-----------------Теоретическая НКФ для АР-----------------\n")
    tNKF_listAR = theoretical_normalCorrelation(betasAR, [], normalCorrelation_listAR, m, ar_m, ar_n, M_min, N_min)
    tnkfAR = tNKF_listAR[ar_m][ar_n]

    print("-----------------Эпсилон для нашей модели-----------------\n")
    epsAR = findEps(tNKF_listAR, normalCorrelation_listAR, m, ar_m, ar_n, ar_m, ar_n)[0][0]

    eps_th_list.append(epsAR)
    eps_list.append(eps_listAR[ar_m][ar_n])

    print(f"--------------------Данные модели СС{ma_m, ma_n}--------------------\n")

    # получаем значения параметров альфа и бета
    print("---------------Значения параметров альфа---------------\n")
    alphasMA = findAlphasMA(correlation_listMA, ma_n)

    extraMEMA = getExtraME([], median, ma_m, ma_n)
    print(f"К значениям последовательности СС{ma_m, ma_n} необходимо добавить {extraMEMA}\n")

    # вычисляем теоретическую НКФ для СС
    print("-----------------Теоретическая НКФ для СС-----------------\n")
    tNKF_listMA = theoretical_normalCorrelation([], alphasMA, normalCorrelation_listMA, m, ma_m, ma_n, M_min, N_min)
    tnkfMA = tNKF_listMA[ma_m][ma_n]

    print("-----------------Эпсилон для нашей модели-----------------\n")
    epsMA = findEps(tNKF_listMA, normalCorrelation_listMA, m, ma_m, ma_n, ma_m, ma_n)[0][0]

    eps_th_list.append(epsMA)
    eps_list.append(eps_listMA[ma_m][ma_n])

    print(f"-------------------Данные модели АРСС{arma_m, arma_n}-------------------\n")

    # находим коэффициенты бета и альфа для каждой модели АРСС
    print("------------Значения параметров альфа и бета------------\n")
    alphasARMA, betasARMA = findBetasAlphasARMA(correlation_listARMA, arma_m, arma_n)

    extraMEARMA = getExtraME(betasARMA, median, arma_m, arma_n)
    print(f"К значениям последовательности АРСС{arma_m, arma_n} необходимо добавить {extraMEARMA}\n")

    # вычисляем теоретическую НКФ для АРСС
    print("----------------Теоретическая НКФ для АРСС----------------\n")
    tNKF_listARMA = theoretical_normalCorrelation(betasARMA, alphasARMA, normalCorrelation_listARMA, m, arma_m, arma_n,
                                                  1, 1)

    tnkfARMA = tNKF_listARMA[arma_m][arma_n]

    # находим эпсилон для каждой модели
    print("-----------------Эпсилон для нашей модели-----------------\n")
    epsARMA = findEps(tNKF_listARMA, normalCorrelation_listARMA, m, arma_m, arma_n, arma_m, arma_n)[0][0]

    eps_th_list.append(epsARMA)
    eps_list.append(eps_listARMA[arma_m][arma_n])

    # строим графики для каждой модели
    nkf_list = [normalCorrelation_listAR, normalCorrelation_listMA, normalCorrelation_listARMA]
    tnkf_list = [tnkfAR, tnkfMA, tnkfARMA]
    name_list = [f"АР({ar_m})", f"СС({ma_n})", f"АРСС{arma_m, arma_n}"]

    for i in range(3):
        plot3(normalCorrelation_list, nkf_list[i], tnkf_list[i], m, name_list[i])

    print("-------------------------Задание №6-------------------------\n")

    # выявляем лучшую модель
    bests = [(ar_m, ar_n), (ma_m, ma_n), (arma_m, arma_n)]
    best_eps, best_m, best_n, best_index = best_of_the_best(eps_th_list, eps_list, bests)

    # получаем процесс, для которого будем строить график
    process_list = [ar, ma, arma]
    median_list = [medianAR, medianMA, medianARMA]
    disp_list = [dispAR, dispMA, dispARMA]

    best_process = process_list[best_index]
    best_median = median_list[best_index]
    best_disp = disp_list[best_index]
    best_name = name_list[best_index]

    # изображаем фрагмент смоделированного СП
    plot1(best_process, best_median, best_disp, f"{best_name} process",
          f"Фрагмент сгенерированного СП по модели {best_name}")


if __name__ == "__main__":
    main()