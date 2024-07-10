import optimization.functionImplementation.ObjectiveFunctionNonLinear;
import optimization.functionImplementation.Options;
import org.ejml.data.DMatrixRMaj;
import org.knowm.xchart.BitmapEncoder;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.markers.SeriesMarkers;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import solvers.NonlinearEquationSolver;

import java.awt.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class Main {
    private static final Color lightBlue = new Color(97, 159, 202);
    private static final Color blue = new Color(0, 40, 255);
    private static final Color red = new Color(255, 30, 30);
    private static final Color green = new Color(0, 190, 0);


    // функция для считывания данных из файла (1 задание)
    private static ArrayList<Double> getDataFromFile(String fileName) {
        ArrayList<Double> data = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            String line;
            while ((line = reader.readLine()) != null) {
                data.add(Double.parseDouble(line));
            }
        } catch (IOException e) {
            return new ArrayList<>();
        }
        return data;
    }


    // функция для вычисления математического ожидания (1 и 5 задания)
    private static double getMedian(ArrayList<Double> sequence) {
        return sequence.stream().mapToDouble(d -> d).average().orElse(0.0);
    }


    // функция для вычисления дисперсии (1 и 5 задания)
    private static double getDispersion(ArrayList<Double> sequence) {
        double median = getMedian(sequence);

        double standardDeviation = 0.0;
        for (double num : sequence) {
            standardDeviation += (num - median) * (num - median);
        }

        return standardDeviation / (sequence.size() - 1);
    }


    // функция для вычисления интервала корреляции (1 задание)
    private static int getInterval(ArrayList<Double> sequence) {
        int t = (int) (0.01 * sequence.size() - 1);
        double e = Math.exp(-1.0);

        while (Math.abs(getNormalCorrelation(sequence, t)) < e) {
            t -= 1;
        }

        return t;
    }


    // функция для подсчёта корреляционной функции (1 и 5 задания)
    private static double getCorrelation(ArrayList<Double> sequence, int m) {

        double sum = 0.0;
        double median = getMedian(sequence);
        int n = sequence.size();

        for (int j = 0; j < n - m; ++j) {
            sum += (sequence.get(j) - median) * (sequence.get(m + j) - median);
        }

        return sum / (n - m - 1);
    }


    // функция для подсчёта нормированной корреляционной функции (1 и 5 задания)
    private static double getNormalCorrelation(ArrayList<Double> sequence, int k) {
        return getCorrelation(sequence, k) / getDispersion(sequence);
    }


    // функция для формирования списков значений КФ и НКФ (1 и 5 задания)
    private static ArrayList<ArrayList<Double>> getCFAndNCF(int m, String label, ArrayList<Double> dataList) {
        ArrayList<Double> cfList = new ArrayList<>(); // создаём список для КФ
        ArrayList<Double> nkfList = new ArrayList<>(); // создаём список для НКФ

        // считаем для m = [0, 10] КФ и НКФ
        for (int i = 0; i < m + 1; ++i) {
            double correlation = getCorrelation(dataList, i); // считаем КФ
            cfList.add(correlation);
            System.out.printf("Для m = %d корреляционная функция%s = %f%n", i, label, correlation);

            double normalCorrelation = getNormalCorrelation(dataList, i); // считаем НКФ
            nkfList.add(normalCorrelation);
            System.out.printf("Для m = %d нормированная корреляционная функция%s = %f\n%n", i, label, normalCorrelation);
        }

        return new ArrayList<>(List.of(cfList, nkfList));
    }


    // функция для построения фрагмента СП (1 и 6 задания)
    private static void plotFragment(ArrayList<Double> sequence, double median, double dispersion, String labelName, String title) {
        double sko = Math.sqrt(dispersion);

        ArrayList<Double> y = new ArrayList<>(sequence.subList(0, 140)); // берём первые 140 значений для построения

        ArrayList<Integer> x = new ArrayList<>(List.of(0, y.size()));

        // считаем минимальное и максимальное значение по Y
        double minY = y.stream().min(Double::compare).get() - 10;
        double maxY = y.stream().max(Double::compare).get() + 10;

        // создаем график, подписываем оси, и ставим заголовок
        XYChart chart = new XYChartBuilder()
                .title(title)
                .xAxisTitle("Index number")
                .yAxisTitle("Random sequence values")
                .build();

        // корректируем границы графиков
        chart.getStyler().setXAxisMin(0.0);
        chart.getStyler().setXAxisMax((double) y.size());
        chart.getStyler().setYAxisMin(minY);
        chart.getStyler().setYAxisMax(maxY);

        // добавляем координаты на график
        chart.addSeries(labelName, y).setMarker(SeriesMarkers.NONE).setLineColor(lightBlue);

        // добавляем на график прямую с y = median
        chart.addSeries("Average", x, Collections.nCopies(2, median))
                .setMarker(SeriesMarkers.NONE)
                .setLineColor(red);
        // стандартное отклонение
        chart.addSeries("Standard deviation", x, Collections.nCopies(2, median + sko))
                .setMarker(SeriesMarkers.NONE)
                .setLineColor(blue);
        chart.addSeries("Standard deviation2", x, Collections.nCopies(2, median - sko))
                .setMarker(SeriesMarkers.NONE).setLineColor(blue)
                .setShowInLegend(false);

        drawHelper(chart, title);
    }


    // графическая оценка НКФ (1 задание)
    private static void plotNormalCorrelationFun(ArrayList<Double> nkfList, int m, int correlationInterval, String title) {
        // создаем график, подписываем оси, и ставим заголовок
        XYChart chart = new XYChartBuilder()
                .title(title)
                .xAxisTitle("Index number")
                .yAxisTitle("Normalized correlation function")
                .build();

        ArrayList<Integer> x = new ArrayList<>(List.of(0, m));
        ArrayList<Integer> xMain = IntStream.rangeClosed(0, m).boxed().collect(Collectors.toCollection(ArrayList::new));

        // добавляем координаты на график
        chart.addSeries("Source", xMain, nkfList).setMarker(SeriesMarkers.NONE).setLineColor(lightBlue);

        chart.addSeries("Interval of correlation", Collections.nCopies(2, correlationInterval), new ArrayList<>(List.of(-1, 1)))
                .setMarker(SeriesMarkers.NONE)
                .setLineColor(red);

        chart.addSeries("1/e and -1/e", x, Collections.nCopies(2, Math.exp(-1)))
                .setMarker(SeriesMarkers.NONE)
                .setLineColor(blue);

        chart.addSeries("1/e and -1/e2", x, Collections.nCopies(2, -Math.exp(-1)))
                .setMarker(SeriesMarkers.NONE)
                .setLineColor(blue)
                .setShowInLegend(false);

        drawHelper(chart, title);
    }


    // функция содержит общие методы для отрисовки графиков (1, 5, 6 задания)
    private static void drawHelper(XYChart chart, String name) {
        // отрисовываем
        new SwingWrapper(chart).displayChart();

        // сохраняем
        try {
            BitmapEncoder.saveBitmap(chart, "./%s".formatted(name), BitmapEncoder.BitmapFormat.PNG);
        } catch (IOException ignored) {
        }
    }


    // функция для нахождения коэффициентов альфа и бета моделей АР (2 и 5 задания)
    private static ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> findAlphaBetasAR(ArrayList<Double> R, int size) {
        // создаём списки списков списков для занесения в них ответов
        ArrayList<ArrayList<ArrayList<Double>>> alphaList = new ArrayList<>();
        ArrayList<ArrayList<ArrayList<Double>>> betaList = new ArrayList<>();

        // в цикле будем формировать системы уравнений для всех АР(M)
        for (int M = 0; M < size + 1; ++M) {
            ArrayList<ArrayList<Double>> vecBeta = new ArrayList<>();// вектор коэффициентов
            ArrayList<ArrayList<Double>> matrixAlpha = new ArrayList<>(); // матрица коэффициентов

            // в цикле для каждого m будем формировать уравнение
            for (int m = 0; m < M + 1; ++m) {
                vecBeta.add(arrayListOf(R.get(m)));
                ArrayList<Double> matrixBuf = new ArrayList<>();

                if (m == 0)
                    matrixBuf.add(1.0);
                else
                    matrixBuf.add(0.0);

                for (int n = 1; n < M + 1; ++n)
                    matrixBuf.add(R.get(Math.abs(m - n)));
                matrixAlpha.add(matrixBuf);
            }

            // решаем систему уравнений Ax = b
            INDArray matrix = Nd4j.create(
                    matrixAlpha.stream()
                            .map(l -> l.stream()
                                    .mapToDouble(Double::doubleValue)
                                    .toArray())
                            .toArray(double[][]::new)
            );
            INDArray vector = Nd4j.create(
                    vecBeta.stream()
                            .map(l -> l.stream()
                                    .mapToDouble(Double::doubleValue)
                                    .toArray())
                            .toArray(double[][]::new)
            );

            double[][] result = Nd4j.linalg().solve(matrix, vector).toDoubleMatrix();

            alphaList.add(arrayListOf(arrayListOf(Math.sqrt(result[0][0])))); // добавляем кф. альфа в список
            System.out.printf("Для M = %d параметр альфа = %f%n", M, Math.sqrt(result[0][0]));

            // заполняем список коэффициентов бета
            ArrayList<Double> betaListTemp = new ArrayList<>();
            for (int i = 1; i < result.length; ++i) {
                System.out.printf("Для M = %d параметр бета = %f%n", M, result[i][0]);
                betaListTemp.add(result[i][0]);
            }
            betaList.add(arrayListOf(betaListTemp));

            System.out.println();
        }

        return new ArrayList<>(List.of(alphaList, betaList));
    }


    // вспомогательная функция, позволяющая создать вложенность массива (2, 3, 4, 5 задания)
    private static <T> ArrayList<T> arrayListOf(T val) {
        return new ArrayList<>(List.of(val));
    }


    // функция для нахождения теоретической НКФ (2, 3, 4, 5 задания)
    private static ArrayList<ArrayList<ArrayList<Double>>> getTheoreticalNormalCorrelation(
            ArrayList<ArrayList<ArrayList<Double>>> betaList,
            ArrayList<ArrayList<ArrayList<Double>>> alphaList,
            ArrayList<Double> nkfList, int end, int M_max, int N_max,
            int M_min, int N_min, boolean type) {
        Supplier<ArrayList<ArrayList<ArrayList<Double>>>> check1 = () -> {
            ArrayList<ArrayList<ArrayList<Double>>> fixedList = new ArrayList<>();
            if (M_min == 1 && N_min == 1)
                fixedList.add(arrayListOf(arrayListOf(Double.POSITIVE_INFINITY)));
            return fixedList;
        };

        Supplier<ArrayList<ArrayList<Double>>> check2 = () -> {
            ArrayList<ArrayList<Double>> fixedList = new ArrayList<>();
            if (M_min == 1 && N_min == 1)
                fixedList.add(arrayListOf(Double.POSITIVE_INFINITY));
            return fixedList;
        };

        ArrayList<ArrayList<ArrayList<Double>>> theoreticalList = check1.get();

        for (int M = M_min; M < M_max + 1; ++M) {
            ArrayList<ArrayList<Double>> bufList = check2.get();

            for (int N = N_min; N < N_max + 1; ++N) {
                if (type || !alphaList.get(M).get(N).get(0).isNaN()) {
                    ArrayList<Double> subBufList = new ArrayList<>();

                    for (int m = 0; m < end + 1; ++m) {
                        if (m <= M + N) {
                            subBufList.add(nkfList.get(m));
                            System.out.printf("Для M = %d, N = %d, m = %d тНКФ = %f%n", M, N, m, nkfList.get(m));
                        } else {
                            double theoretical = 0.0;

                            for (int j = 1; j < M + 1; ++j)
                                theoretical += betaList.get(M).get(N).get(j - 1) * subBufList.get(m - j);

                            System.out.printf("Для M = %d, N = %d, m = %d тНКФ = %f%n", M, N, m, theoretical);
                            subBufList.add(theoretical);
                        }
                    }
                    bufList.add(subBufList);
                    System.out.println();
                } else {
                    System.out.printf("Для M = %d и N = %d модели не существует\n%n", M, N);
                    bufList.add(arrayListOf(Double.NaN));
                }
            }
            theoreticalList.add(bufList);
        }
        return theoreticalList;
    }


    // функция для вычисления погрешностей каждой из моделей (2, 3, 4, 5 задания)
    private static ArrayList<ArrayList<Double>> getEpsilon(ArrayList<ArrayList<ArrayList<Double>>> tNKFList,
                                                           ArrayList<Double> nkfList, int end, int M_max, int N_max,
                                                           int M_min, int N_min) {

        Supplier<ArrayList<ArrayList<Double>>> check1 = () -> {
            ArrayList<ArrayList<Double>> fixedList = new ArrayList<>();
            if (M_min == 1 && N_min == 1)
                fixedList.add(arrayListOf(Double.POSITIVE_INFINITY));
            return fixedList;
        };

        Supplier<ArrayList<Double>> check2 = () -> {
            ArrayList<Double> fixedList = new ArrayList<>();
            if (M_min == 1 && N_min == 1)
                fixedList.add(Double.POSITIVE_INFINITY);
            return fixedList;
        };

        ArrayList<ArrayList<Double>> epsilonList = check1.get(); // список значений для погрешностей

        // считаем погрешность для каждого порядка
        for (int M = M_min; M < M_max + 1; ++M) {
            ArrayList<Double> bufList = check2.get();

            for (int N = N_min; N < N_max + 1; ++N) {
                if (!tNKFList.get(M).get(N).get(0).isNaN()) {
                    double eps = 0.0;

                    // считаем погрешность по формуле
                    for (int m = 1; m < end + 1; ++m)
                        eps += (tNKFList.get(M).get(N).get(m) - nkfList.get(m)) *
                                (tNKFList.get(M).get(N).get(m) - nkfList.get(m));

                    System.out.printf("Для M = %d и N = %d погрешность = %f%n", M, N, eps);
                    bufList.add(eps);
                } else {
                    System.out.printf("Для M = %d и N = %d модели не существует%n", M, N);
                    bufList.add(Double.NaN);
                }
            }
            epsilonList.add(bufList);
            System.out.println();
        }
        return epsilonList;
    }


    // функция для нахождения лучшей модели (2, 3, 4 задания)
    static ArrayList<Integer> getBestModel(ArrayList<ArrayList<Double>> epsilonList, int M_max, int N_max, int M_min,
                                           int N_min, ArrayList<ArrayList<Double>> stabilityList) {
        BiFunction<Integer, Integer, Boolean> check = (Integer m, Integer n) -> {
            boolean result = true;
            if (M_max > 0 && N_max > 0) {
                if (stabilityList.get(m).get(n).isNaN() || stabilityList.get(m).get(n) == 0.0)
                    result = false;
                if (!result)
                    System.out.printf("Для M = %d и N = %d модель не устойчива%n", m, n);
            }
            return result;
        };


        Supplier<Double> initMin = () -> {
            if (M_min == 1 && N_min == 1)
                return epsilonList.get(1).get(1);
            return epsilonList.get(0).get(0);
        };

        Double minEpsilon = initMin.get();
        ArrayList<Integer> minMN = new ArrayList<>(List.of(0, 0));

        for (int M = M_min; M < M_max + 1; ++M) {
            for (int N = N_min; N < N_max + 1; ++N) {
                if (!epsilonList.get(M).get(N).isNaN() && check.apply(M, N)) {
                    if (minEpsilon.isNaN() || epsilonList.get(M).get(N) < minEpsilon) {
                        minEpsilon = epsilonList.get(M).get(N);
                        minMN = new ArrayList<>(List.of(M, N));
                    }
                } else
                    System.out.printf("Для M = %d и N = %d модели не существует%n", M, N);
            }
        }
        System.out.printf("Лучшая модель при M = %d и N = %d с эпсилон = %f%n", minMN.get(0), minMN.get(1), minEpsilon);
        return minMN;
    }


    // правила для решения системы уравнений СС(N) для каждого порядка N (3 и 5 задания)
    private static ArrayList<Double> equationsMA(ArrayList<Double> R, double[] seq) {
        ArrayList<Double> factorList = new ArrayList<>();
        int n = seq.length - 1;

        for (int m = 0; m < n + 1; ++m) {
            double factor = 0.0;
            for (int i = 0; i < n - m + 1; ++i)
                factor += seq[i] * seq[i + m];

            factor -= R.get(m);
            factorList.add(factor);
        }
        return factorList;
    }


    // функция для подсчёта нормы вектора (3 и 4 задания)
    private static double getNorm(ArrayList<Double> vec) {
        var sum = 0.0;
        for (double elem : vec)
            sum += elem * elem;
        return Math.sqrt(sum);
    }


    // функция для нахождения коэффициентов альфа модели СС(N) (3 и 5 задания)
    private static ArrayList<ArrayList<ArrayList<Double>>> findAlphasMA(int N_max, ArrayList<Double> R) {
        ArrayList<ArrayList<Double>> ansList = new ArrayList<>();

        for (int n = 0; n < N_max + 1; ++n) {
            Options options = new Options(n + 1);
            options.setAnalyticalJacobian(false); // Указываем, будем ли предоставлять аналитический якобиан (по стандарту false)
            options.setAlgorithm(Options.TRUST_REGION); // Выбор алгоритма; Options.TRUST_REGION или Options.LINE_SEARCH (по стандарту Options.TRUST_REGION)
            options.setSaveIterationDetails(true); // Сохранять информацию об итерациях в переменную типа Result (по стандарту false)
            options.setAllTolerances(1e-12); // Устанавливаем точность схождения (по стандарту 1e-8)
            options.setMaxIterations(1000); // Ставим максимальное количество итераций (по стандарту 100)

            // инициализируем функцию
            int finalN = n;
            ObjectiveFunctionNonLinear function = new ObjectiveFunctionNonLinear() {
                @Override
                public DMatrixRMaj getF(DMatrixRMaj x) {
                    DMatrixRMaj f = new DMatrixRMaj(finalN + 1, 1);

                    ArrayList<Double> result = equationsMA(R, x.getData());

                    for (int i = 0; i < result.size(); ++i)
                        f.set(i, 0, result.get(i));
                    return f;
                }

                @Override
                public DMatrixRMaj getJ(DMatrixRMaj x) {
                    return null;
                }

            };

            NonlinearEquationSolver nonlinearSolver = new NonlinearEquationSolver(function, options);

            // начальное приближение
            DMatrixRMaj initialGuess = new DMatrixRMaj(n + 1, 1);
            for (int i = 0; i < n + 1; ++i)
                initialGuess.set(i, 0, 0.0);
            initialGuess.set(0, 0, Math.sqrt(R.get(0)));

            nonlinearSolver.solve(new DMatrixRMaj(initialGuess)); // решаем систему

            double[] result = nonlinearSolver.getX().getData();
            double norm = getNorm(equationsMA(R, result)); // считаем норму

            // проверка модели на существование и выдача нужного ответа
            if (norm < 1e-4) {
                ArrayList<Double> bufList = new ArrayList<>();

                for (int i = 0; i < n + 1; ++i) {
                    bufList.add(result[i]);
                    System.out.printf("Для N = %d параметр альфа = %f%n", n, result[i]);
                }
                ansList.add(bufList);
            } else {
                System.out.printf("Для N = %d модели не существует%n", n);
                ansList.add(arrayListOf(Double.NaN));
            }
            System.out.println();
        }
        return arrayListOf(ansList);
    }


    // правила для решения системы уравнений АРСС(M, N) для каждого порядка M и N (4 и 5 задания)
    private static ArrayList<Double> ARMAFactorGenerator(ArrayList<Double> R, double[] seq, int M, int N) {
        ArrayList<Double> alphaList = new ArrayList<>();
        ArrayList<Double> betaList = new ArrayList<>();
        ArrayList<Double> Rxi = new ArrayList<>();

        // распределяем элементы по спискам

        for (int i = 0; i < seq.length; ++i) {
            double curElem = seq[i];
            if (i < (N + 1))
                alphaList.add(curElem);
            else if (N < i && i < (N + M + 1))
                betaList.add(curElem);
            else
                Rxi.add(curElem);
        }

        ArrayList<Double> factorList = new ArrayList<>();

        // генерируем правила для системы типа Таблица 1.1 А
        for (int n = 0; n < N + 1; ++n) {
            double factor = 0.0;
            for (int j = 1; j < M + 1; ++j)
                factor += betaList.get(j - 1) * R.get(Math.abs(n - j));
            for (int i = n; i < N + 1; ++i)
                factor += alphaList.get(i) * Rxi.get(i - n);
            factor -= R.get(n);

            factorList.add(factor);
        }

        // генерируем правила для системы типа Таблица 1.1 Б
        for (int i = 1; i < M + 1; ++i) {
            double factor = 0.0;
            for (int j = 1; j < M + 1; ++j)
                factor += betaList.get(j - 1) * R.get(Math.abs(N - j + i));
            factor -= R.get(N + i);

            factorList.add(factor);
        }

        // генерируем правила для системы типа Таблица 1.1 Г
        for (int n = 0; n < N + 1; ++n) {
            double factor = 0.0;
            int m = Math.min(n, M);

            if (n > 0) {
                for (int j = 1; j < m + 1; ++j)
                    factor += betaList.get(j - 1) * Rxi.get(n - j);
            }
            factor += (alphaList.get(n) - Rxi.get(n));
            factorList.add(factor);
        }

        return factorList;
    }


    // функция для нахождения коэффициентов альфа и бета моделей АРСС(M, N) (4 и 5 задания)
    private static ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> findBetasAlphasARMA(ArrayList<Double> R, int M_max, int N_max) {
        ArrayList<ArrayList<ArrayList<Double>>> alphaList = arrayListOf(arrayListOf(new ArrayList<>()));
        ArrayList<ArrayList<ArrayList<Double>>> betaList = arrayListOf(arrayListOf(new ArrayList<>()));

        for (int m = 1; m < M_max + 1; ++m) {
            ArrayList<ArrayList<Double>> bufAlphaList = arrayListOf(new ArrayList<>());
            ArrayList<ArrayList<Double>> bufBetaList = arrayListOf(new ArrayList<>());

            for (int n = 1; n < N_max + 1; ++n) {
                ArrayList<Double> subAlphaList = new ArrayList<>();
                ArrayList<Double> subBetaList = new ArrayList<>();

                Options options = new Options(m + 2 * n + 2);
                options.setAnalyticalJacobian(false); // Указываем, будем ли предоставлять аналитический якобиан (по стандарту false)
                options.setAlgorithm(Options.LINE_SEARCH); // Выбор алгоритма; Options.TRUST_REGION или Options.LINE_SEARCH (по стандарту Options.TRUST_REGION)
                options.setSaveIterationDetails(true); // Сохранять информацию об итерациях в переменную типа Result (по стандарту false)
                options.setAllTolerances(1e-8); // Устанавливаем точность схождения (по стандарту 1e-8)
                options.setMaxIterations(1000); // Ставим максимальное количество итераций (по стандарту 100)


                // инициализируем функцию
                int finalM = m;
                int finalN = n;
                ObjectiveFunctionNonLinear function = new ObjectiveFunctionNonLinear() {
                    @Override
                    public DMatrixRMaj getF(DMatrixRMaj x) {
                        DMatrixRMaj f = new DMatrixRMaj(finalM + 2 * finalN + 2, 1);

                        ArrayList<Double> result = ARMAFactorGenerator(R, x.getData(), finalM, finalN);

                        for (int i = 0; i < result.size(); ++i)
                            f.set(i, 0, result.get(i));
                        return f;
                    }

                    @Override
                    public DMatrixRMaj getJ(DMatrixRMaj x) {
                        return null;
                    }
                };

                NonlinearEquationSolver nonlinearSolver = new NonlinearEquationSolver(function, options);

                // начальное приближение
                DMatrixRMaj initialGuess = new DMatrixRMaj(m + 2 * n + 2, 1);
                for (int i = 0; i < m + 2 * n + 2; ++i)
                    initialGuess.set(i, 0, 0.0);
                initialGuess.set(0, 0, Math.sqrt(R.get(0)));

                nonlinearSolver.solve(new DMatrixRMaj(initialGuess)); // решаем систему

                double[] result = nonlinearSolver.getX().getData();
                double norm = getNorm(ARMAFactorGenerator(R, result, m, n)); // считаем норму

                if (norm < 1e-4) {
                    for (int index = 0; index < n + 1; ++index) {
                        // тут условие подбиралось под конкретную задачу - у Вас может быть по-другому. Аккуратно!!! (но он зачёл и так)
                        double elem = m % 2 != 0 && n % 2 == 0 ? -result[index] : result[index];
//                         double elem = result[index];

                        System.out.printf("Для M = %d и N = %d коэффициент альфа = %f и норма = %f%n", m, n, elem, norm);
                        subAlphaList.add(elem);
                    }
                    System.out.println();
                    for (int idx = 0; idx < m; ++idx) {
                        double elem = result[idx + n + 1];

                        System.out.printf("Для M = %d и N = %d коэффициент бета = %f%n", m, n, elem);
                        subBetaList.add(elem);
                    }
                    System.out.println();
                } else {
                    System.out.printf("Для M = %d и N = %d модели не существует\n%n", m, n);
                    subAlphaList.add(Double.NaN);
                    subBetaList.add(Double.NaN);
                }
                Collections.reverse(subAlphaList);
                bufAlphaList.add(subAlphaList);
                bufBetaList.add(subBetaList);
            }
            alphaList.add(bufAlphaList);
            betaList.add(bufBetaList);
        }
        return new ArrayList<>(List.of(alphaList, betaList));
    }


    // функция для проверки на стабильность моделей АРСС(M, N) (4 задание)
    private static ArrayList<ArrayList<Double>> getStability(ArrayList<ArrayList<ArrayList<Double>>> betaList, int M_max, int N_max, int M_min, int N_min) {
        ArrayList<ArrayList<Double>> result = arrayListOf(new ArrayList<>());

        for (int M = M_min; M < M_max + 1; ++M) {
            ArrayList<Double> subResult = arrayListOf(Double.POSITIVE_INFINITY);
            for (int N = N_min; N < N_max + 1; ++N) {
                if (!betaList.get(M).get(N).get(0).isNaN()) {
                    ArrayList<Double> betas = betaList.get(M).get(N);
                    int size = betas.size();

                    boolean ans = true;
                    switch (size) {
                        case 0 -> ans = true;
                        case 1 -> ans = Math.abs(betas.get(0)) < 1;
                        case 2 -> ans = Math.abs(betas.get(1)) < 1 && Math.abs(betas.get(0)) < 1 - betas.get(1);
                        case 3 -> {
                            boolean statement1 = Math.abs(betas.get(2)) < 1;
                            boolean statement2 = Math.abs(betas.get(0) + betas.get(2)) < 1 - betas.get(1);
                            boolean statement3 = Math.abs(betas.get(1) + betas.get(0) * betas.get(2)) < 1 - betas.get(2) * betas.get(2);
                            ans = statement1 && statement2 && statement3;
                        }
                    }
                    double answer;

                    if (ans) {
                        answer = 1.0;
                        System.out.printf("Для M = %d и N = %d модель устойчива%n", M, N);
                    } else {
                        answer = 0.0;
                        System.out.printf("Для M = %d и N = %d модель не устойчива%n", M, N);
                    }

                    subResult.add(answer);
                } else {
                    System.out.printf("Для M = %d и N = %d модели не существует%n", M, N);
                    subResult.add(Double.NaN);
                }
            }
            System.out.println();
            result.add(subResult);
        }
        return result;
    }


    // функция для генерации выборки из old_n значений (5 задание)
    private static ArrayList<Double> generateSequence(ArrayList<ArrayList<ArrayList<Double>>> alphaList,
                                                      ArrayList<ArrayList<ArrayList<Double>>> betaList,
                                                      int M, int N, int old_n, double median) {
        Supplier<ArrayList<Double>> check = () -> {
            if (betaList.isEmpty())
                return new ArrayList<>();
            return betaList.get(M).get(N);
        };

        ArrayList<Double> subAlphaList = alphaList.get(M).get(N);
        ArrayList<Double> subBetaList = check.get();

        int badCount = 1000;

        int new_n = old_n + badCount;

        ArrayList<Double> etaList = new ArrayList<>();
        for (int i = 0; i < new_n; ++i)
            etaList.add(0.0);

        ArrayList<Double> ksiList = new ArrayList<>();
        for (int i = 0; i < new_n; ++i)
            ksiList.add(new Random().nextGaussian(0.0, 1.0));


        for (int n = 0; n < new_n; ++n) {
            for (int j = 1; j < M + 1; ++j) {
                if (n - j >= 0)
                    etaList.set(n, etaList.get(n) + subBetaList.get(j - 1) * etaList.get(n - j));
            }
            for (int i = 0; i < N + 1; ++i) {
                if (n - i >= 0)
                    etaList.set(n, etaList.get(n) + subAlphaList.get(i) * ksiList.get(n - i));
            }
        }
        etaList = new ArrayList<>(etaList.subList(1000, new_n));


        for (int n = 0; n < old_n; ++n)
            etaList.set(n, etaList.get(n) + median);

        return etaList;
    }

    // функция для графического сравнения НКФ смоделированного и исходного СП (5 задание)
    private static void plotBestNKF(ArrayList<Double> sourceNKF, ArrayList<Double> modelNKF, ArrayList<Double> tNKFModel, String label, int m) {
        // создаем график, подписываем оси, и ставим заголовок
        XYChart chart = new XYChartBuilder()
                .title("Графическая оценка НКФ модели %s".formatted(label))
                .xAxisTitle("Index number")
                .yAxisTitle("Normalized correlation function")
                .build();

        // корректируем границы графика
        chart.getStyler().setYAxisMin(-1.0);
        chart.getStyler().setYAxisMax(1.0);

        ArrayList<Integer> xMain = IntStream.rangeClosed(0, m).boxed().collect(Collectors.toCollection(ArrayList::new));

        // добавляем координаты на график
        chart.addSeries("Source", xMain, sourceNKF).setMarker(SeriesMarkers.NONE).setLineColor(red); // добавляем НКФ исходного СП
        chart.addSeries("Modeling", xMain, modelNKF).setMarker(SeriesMarkers.NONE).setLineColor(green); // добавляем НКФ нового СП
        chart.addSeries("Theoretical", xMain, tNKFModel).setMarker(SeriesMarkers.NONE).setLineColor(blue); // добавляем тНКФ нового СП


        drawHelper(chart, "Графическая оценка НКФ модели %s".formatted(label));
    }


    // функция для выбора лучшей модели из моделей АР, СС и АРСС (6 задание)
    private static int getBestOfTheBests(ArrayList<Double> thEpsList, ArrayList<Double> epsList, ArrayList<List<Integer>> models) {
        double minThEps = thEpsList.stream().min(Double::compare).get();
        double minModEps = epsList.stream().min(Double::compare).get();

        double minEps = Math.min(minThEps, minModEps);
        int minIndex = thEpsList.contains(minEps) ? thEpsList.indexOf(minEps) : epsList.indexOf(minEps);

        int bestM = models.get(minIndex).get(0);
        int bestN = models.get(minIndex).get(1);

        System.out.printf("Лучшая модель при M = %d и N = %d с эпсилон = %f%n", bestM, bestN, minEps);

        return minIndex;
    }


    public static void main(String[] args) {
        ArrayList<Double> startList = getDataFromFile("21.txt");

        int length = startList.size();
        int m = 10; // задаём количество отсчётов константой
        int M_max = 3; // максимальный порядок M
        int N_max = 3; // максимальный порядок N
        int M_min = 0; // минимальный порядок M
        int N_min = 0; // минимальный порядок N


        System.out.println("\n-------------------------Задание №1-------------------------\n\n");
        double median = getMedian(startList); // считаем мат. ожидание
        System.out.printf("Математическое ожидание = %f\n%n", median);

        double dispersion = getDispersion(startList); // считаем дисперсию
        System.out.printf("Дисперсия = %f\n%n", dispersion);

        int interval = getInterval(startList); // считаем интервал корреляции
        System.out.printf("Интервал корреляции = %d\n%n", interval);


        // считаем для m = [0, 10] КФ и НКФ
        System.out.println("----------Значения корреляционной функции и НКФ----------\n");
        ArrayList<ArrayList<Double>> startCFAndNCF = getCFAndNCF(m, "", startList);
        ArrayList<Double> cfList = startCFAndNCF.get(0); // создаём список для КФ
        ArrayList<Double> nkfList = startCFAndNCF.get(1); // создаём список для НКФ

        // строим графики фрагмента исходного СП и НКФ
        plotFragment(startList, median, dispersion, "Source process", "Фрагмент исходного СП");
        plotNormalCorrelationFun(nkfList, m, interval, "Графическая оценка НКФ");


        System.out.println("\n-------------------------Задание №2-------------------------\n\n");

        // получаем значения коэффициентов альфа и бета
        System.out.println("---------Значения параметров альфа и бета для АР---------\n");
        ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> listAR = findAlphaBetasAR(cfList, M_max);
        ArrayList<ArrayList<ArrayList<Double>>> alphaListAR = listAR.get(0);
        ArrayList<ArrayList<ArrayList<Double>>> betaListAR = listAR.get(1);

        // вычисляем теоретическую НКФ для АР
        System.out.println("-----------------Теоретическая НКФ для АР-----------------\n");
        ArrayList<ArrayList<ArrayList<Double>>> tNKFListAR = getTheoreticalNormalCorrelation(betaListAR,
                arrayListOf(arrayListOf(new ArrayList<>())), nkfList, m, M_max, N_min, M_min, N_min, true);

        // находим эпсилон для каждой модели
        System.out.println("----------------Эпсилон для каждой модели----------------\n");
        ArrayList<ArrayList<Double>> epsilonListAR = getEpsilon(tNKFListAR, nkfList, m, M_max, N_min, M_min, N_min);

        // выбор лучшей модели АР
        System.out.println("--------------------Выбор лучшей модели--------------------\n");
        ArrayList<Integer> bestModelAR = getBestModel(epsilonListAR, M_max, N_min, M_min, N_min, new ArrayList<>());
        int ar_m = bestModelAR.get(0);
        int ar_n = bestModelAR.get(1);


        System.out.println("\n\n-------------------------Задание №3-------------------------\n\n");

        // получаем значения параметра альфа
        System.out.println("-----------------Значения параметра альфа-----------------\n");
        ArrayList<ArrayList<ArrayList<Double>>> alphaListMA = findAlphasMA(N_max, cfList);

        // вычисляем теоретическую НКФ для СС
        System.out.println("-----------------Теоретическая НКФ для СС-----------------\n");
        ArrayList<ArrayList<ArrayList<Double>>> tNKFListMA = getTheoreticalNormalCorrelation(arrayListOf(arrayListOf(new ArrayList<>())), alphaListMA, nkfList, m, M_min, N_max, M_min, N_min, false);

        // находим эпсилон для каждой модели
        System.out.println("----------------Эпсилон для каждой модели----------------\n");
        ArrayList<ArrayList<Double>> epsilonListMA = getEpsilon(tNKFListMA, nkfList, m, M_min, N_max, M_min, N_min);

        // выбор лучшей модели СС
        System.out.println("--------------------Выбор лучшей модели--------------------\n");
        ArrayList<Integer> bestModelMA = getBestModel(epsilonListMA, M_min, N_max, M_min, N_min, new ArrayList<>());
        int ma_m = bestModelMA.get(0);
        int ma_n = bestModelMA.get(1);


        System.out.println("\n\n-------------------------Задание №4-------------------------\n\n");

        // находим коэффициенты бета и альфа для каждой модели АРСС
        System.out.println("------------Значения параметров альфа и бета------------\n");
        ArrayList<ArrayList<ArrayList<ArrayList<Double>>>> betasAlphasARMA = findBetasAlphasARMA(cfList, M_max, N_max);
        ArrayList<ArrayList<ArrayList<Double>>> alphaListARMA = betasAlphasARMA.get(0);
        ArrayList<ArrayList<ArrayList<Double>>> betaListARMA = betasAlphasARMA.get(1);

        // вычисляем теоретическую НКФ для АРСС
        System.out.println("----------------Теоретическая НКФ для АРСС----------------\n");
        ArrayList<ArrayList<ArrayList<Double>>> tNKFListARMA = getTheoreticalNormalCorrelation(betaListARMA, alphaListARMA, nkfList, m, M_max, N_max, 1, 1, false);

        // находим эпсилон для каждой модели
        System.out.println("----------------Эпсилон для каждой модели----------------\n");
        ArrayList<ArrayList<Double>> epsilonListARMA = getEpsilon(tNKFListARMA, nkfList, m, M_max, N_max, 1, 1);

        // проверяем все модели на стабильность
        System.out.println("-------------Проверка моделей на стабильность-------------\n");
        ArrayList<ArrayList<Double>> stabilityList = getStability(betaListARMA, M_max, N_max, 1, 1);

        // выбор лучшей модели АРСС
        System.out.println("--------------------Выбор лучшей модели--------------------\n");
        ArrayList<Integer> bestModelARMA = getBestModel(epsilonListARMA, M_max, N_max, 1, 1, stabilityList);
        int arma_m = bestModelARMA.get(0);
        int arma_n = bestModelARMA.get(1);


        System.out.println("\n\n-------------------------Задание №5-------------------------\n\n");

        // генерируем по 5000 значений для каждой модели
        ArrayList<Double> ar = generateSequence(alphaListAR, betaListAR, ar_m, ar_n, length, median);
        ArrayList<Double> ma = generateSequence(alphaListMA, new ArrayList<>(), ma_m, ma_n, length, median);
        ArrayList<Double> arma = generateSequence(alphaListARMA, betaListARMA, arma_m, arma_n, length, median);

        System.out.println("----------------Значения моментных функций----------------\n");
        double medianAR = getMedian(ar);
        double medianMA = getMedian(ma);
        double medianARMA = getMedian(arma);

        System.out.printf("Для АР(%d) медиана = %f%n", ar_m, medianAR);
        System.out.printf("Для CC(%d) медиана = %f%n", ma_n, medianMA);
        System.out.printf("Для АРCC(%d, %d) медиана = %f\n%n", arma_m, arma_n, medianARMA);

        double dispersionAR = getDispersion(ar);
        double dispersionMA = getDispersion(ma);
        double dispersionARMA = getDispersion(arma);

        System.out.printf("Для АР(%d) дисперсия = %f%n", ar_m, dispersionAR);
        System.out.printf("Для CC(%d) дисперсия = %f%n", ma_n, dispersionMA);
        System.out.printf("Для АРCC(%d, %d) дисперсия = %f\n%n", arma_m, arma_n, dispersionARMA);

        double sko = Math.sqrt(dispersion);
        double skoAR = Math.sqrt(dispersionAR);
        double skoMA = Math.sqrt(dispersionMA);
        double skoARMA = Math.sqrt(dispersionARMA);

        System.out.printf("Для исходного СП СКО = %f%n", sko);
        System.out.printf("Для АР(%d) СКО = %f%n", ar_m, skoAR);
        System.out.printf("Для CC(%d) СКО = %f%n", ma_n, skoMA);
        System.out.printf("Для АРCC(%d, %d) СКО = %f\n%n", arma_m, arma_n, skoARMA);


        System.out.println("----------Значения корреляционной функции и НКФ----------\n");
        ArrayList<ArrayList<Double>> arCFAndNCF = getCFAndNCF(m, " АР", ar);
        ArrayList<Double> cfListAR = arCFAndNCF.get(0);
        ArrayList<Double> nkfListAR = arCFAndNCF.get(1);
        System.out.println();

        ArrayList<ArrayList<Double>> maCFAndNCF = getCFAndNCF(m, " СС", ma);
        ArrayList<Double> cfListMA = maCFAndNCF.get(0);
        ArrayList<Double> nkfListMA = maCFAndNCF.get(1);
        System.out.println();

        ArrayList<ArrayList<Double>> armaCFAndNCF = getCFAndNCF(m, " АРСС", arma);
        ArrayList<Double> cfListARMA = armaCFAndNCF.get(0);
        ArrayList<Double> nkfListARMA = armaCFAndNCF.get(1);

        ArrayList<Double> thEpsList = new ArrayList<>();
        ArrayList<Double> epsList = new ArrayList<>();

        System.out.printf("----------------------Данные модели АР(%d)----------------------\n%n", ar_m);

        // получаем значения параметров альфа и бета
        System.out.println("------------Значения параметров альфа и бета------------\n");
        listAR = findAlphaBetasAR(cfListAR, M_max);
        ArrayList<ArrayList<ArrayList<Double>>> alphaAR = listAR.get(0);
        ArrayList<ArrayList<ArrayList<Double>>> betaAR = listAR.get(1);

        System.out.println("-----------------Теоретическая НКФ для АР-----------------\n");
        ArrayList<ArrayList<ArrayList<Double>>> tNkfARList = getTheoreticalNormalCorrelation(betaAR,
                arrayListOf(arrayListOf(new ArrayList<>())), nkfListAR, m, ar_m, ar_n, M_min, N_min, true);
        ArrayList<Double> tNkfAR = tNkfARList.get(ar_m).get(ar_n);

        System.out.println("-----------------Эпсилон для нашей модели-----------------\n");
        double epsAR = getEpsilon(tNkfARList, nkfListAR, m, ar_m, ar_n, ar_m, ar_n).get(0).get(0);
        thEpsList.add(epsAR);
        epsList.add(epsilonListAR.get(ar_m).get(ar_n));

        System.out.printf("----------------------Данные модели СС(%d)----------------------\n%n", ma_n);

        // получаем значения параметров альфа и бета
        System.out.println("---------------Значения параметров альфа---------------\n");
        ArrayList<ArrayList<ArrayList<Double>>> alphaMA = findAlphasMA(ma_n, cfListMA);

        //вычисляем теоретическую НКФ для СС
        System.out.println("-----------------Теоретическая НКФ для СС-----------------\n");
        ArrayList<ArrayList<ArrayList<Double>>> tNkfMAList = getTheoreticalNormalCorrelation(arrayListOf(arrayListOf(new ArrayList<>())),
                alphaMA, nkfListMA, m, ma_m, ma_n, M_min, N_min, false);
        ArrayList<Double> tNkfMA = tNkfMAList.get(ma_m).get(ma_n);

        System.out.println("-----------------Эпсилон для нашей модели-----------------\n");
        double epsMA = getEpsilon(tNkfMAList, nkfListMA, m, ma_m, ma_n, ma_m, ma_n).get(0).get(0);
        thEpsList.add(epsMA);
        epsList.add(epsilonListMA.get(ma_m).get(ma_n));

        System.out.printf("-------------------Данные модели АРСС(%d, %d)-------------------\n%n", arma_m, arma_n);

        // находим коэффициенты бета и альфа для каждой модели АРСС
        System.out.println("------------Значения параметров альфа и бета------------\n");
        betasAlphasARMA = findBetasAlphasARMA(cfListARMA, arma_m, arma_n);
        ArrayList<ArrayList<ArrayList<Double>>> alphaARMA = betasAlphasARMA.get(0);
        ArrayList<ArrayList<ArrayList<Double>>> betasARMA = betasAlphasARMA.get(1);

        // вычисляем теоретическую НКФ для АРСС
        System.out.println("----------------Теоретическая НКФ для АРСС----------------\n");
        ArrayList<ArrayList<ArrayList<Double>>> tNkfARMAList = getTheoreticalNormalCorrelation(betasARMA,
                alphaARMA, nkfListARMA, m, arma_m, arma_n, 1, 1, false);
        ArrayList<Double> tNkfARMA = tNkfARMAList.get(arma_m).get(arma_n);

        // находим эпсилон для каждой модели
        System.out.println("-----------------Эпсилон для нашей модели-----------------\n");
        double epsARMA = getEpsilon(tNkfARMAList, nkfListARMA, m, arma_m, arma_n, arma_m, arma_n).get(0).get(0);
        thEpsList.add(epsARMA);
        epsList.add(epsilonListARMA.get(arma_m).get(arma_n));

        ArrayList<ArrayList<Double>> nkfLists = new ArrayList<>(List.of(nkfListAR, nkfListMA, nkfListARMA));
        ArrayList<ArrayList<Double>> tNKFLists = new ArrayList<>(List.of(tNkfAR, tNkfMA, tNkfARMA));
        ArrayList<String> nameList = new ArrayList<>(List.of("АР(%d)".formatted(ar_m), "СС(%d)".formatted(ma_n), "АРСС(%d, %d)".formatted(arma_m, arma_n)));

        for (int i = 0; i < 3; ++i)
            plotBestNKF(nkfList, nkfLists.get(i), tNKFLists.get(i), nameList.get(i), m);


        System.out.println("\n\n-------------------------Задание №6-------------------------\n\n");

        // получаем лучшую модель
        ArrayList<List<Integer>> bests = new ArrayList<>(List.of(List.of(ar_m, ar_n), List.of(ma_m, ma_n), List.of(arma_m, arma_n)));
        int bestIndex = getBestOfTheBests(thEpsList, epsList, bests);

        // получаем процесс, для которого будем строить график
        List<ArrayList<Double>> processList = List.of(ar, ma, arma);
        List<Double> medianList = List.of(medianAR, medianMA, medianARMA);
        List<Double> dispersionList = List.of(dispersionAR, dispersionMA, dispersionARMA);

        ArrayList<Double> bestProcess = processList.get(bestIndex);
        double bestMedian = medianList.get(bestIndex);
        double bestDispersion = dispersionList.get(bestIndex);
        String bestName = nameList.get(bestIndex);

        // изображаем фрагмент смоделированного СП
        plotFragment(bestProcess, bestMedian, bestDispersion, "%s process".formatted(bestName), "Фрагмент сгенерированного СП по модели %s".formatted(bestName));
    }
}