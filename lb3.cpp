#include <iostream>
#include <vector>
#include <mpi.h>
#include <chrono>
#include <fstream>
#include <string>

using namespace std;
using namespace chrono;


bool readMatrix(const string& filename, vector<double>& matrix, int& n) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: cannot open file " << filename << endl;
        return false;
    }

    file >> n;
    matrix.resize(n * n);
    for (int i = 0; i < n * n; i++) {
        file >> matrix[i];
    }
    file.close();
    return true;
}

// Функция для записи матрицы в файл
void writeMatrix(const string& filename, const vector<double>& matrix, int n) {
    ofstream file(filename);
    file << n << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << matrix[i * n + j] << " ";
        }
        file << endl;
    }
    file.close();
}

int main(int argc, char* argv[]) {
    int rank, size;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Размер матрицы (можно передать как аргумент командной строки)
    int N = (argc > 1) ? atoi(argv[1]) : 400;

    // Векторы для хранения матриц
    vector<double> A, B, C;
    int matrix_size = N * N;

    // Только процесс 0 читает матрицы из файлов
    if (rank == 0) {
        int n_check;
        if (!readMatrix("matrix_A_" + to_string(N) + ".txt", A, n_check)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (!readMatrix("matrix_B_" + to_string(N) + ".txt", B, n_check)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (n_check != N) {
            cerr << "Matrix size mismatch!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cout << "Matrices loaded. Size: " << N << "x" << N << endl;
        cout << "Number of processes: " << size << endl;
    }

    // Рассылаем матрицу B всем процессам
    // Сначала выделяем память на всех процессах
    if (rank != 0) {
        B.resize(matrix_size);
    }

    // Широковещательная рассылка матрицы B от процесса 0 всем остальным
    MPI_Bcast(B.data(), matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Определяем, сколько строк будет обрабатывать каждый процесс
    int rows_per_proc = N / size;
    int remainder = N % size;

    // Количество строк для текущего процесса
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    // Вектор для локальной части матрицы A
    vector<double> local_A(local_rows * N);

    // Создаем массивы для описания распределения строк
    vector<int> send_counts(size), displs(size);

    if (rank == 0) {
        // Заполняем информацию о том, сколько строк отправлять каждому процессу
        for (int i = 0; i < size; i++) {
            send_counts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * N;
            displs[i] = (i == 0) ? 0 : displs[i - 1] + send_counts[i - 1];
        }
    }

    // Распределяем строки матрицы A между процессами
    MPI_Scatterv(A.data(), send_counts.data(), displs.data(), MPI_DOUBLE,
        local_A.data(), local_rows * N, MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // Локальное умножение
    vector<double> local_C(local_rows * N, 0.0);

    // Синхронизация перед замером времени
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    // Основной цикл умножения
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += local_A[i * N + k] * B[k * N + j];
            }
            local_C[i * N + j] = sum;
        }
    }

    // Синхронизация после умножения
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();

    // Собираем результаты на процессе 0
    if (rank == 0) {
        C.resize(matrix_size);
    }

    // Собираем все части матрицы C
    MPI_Gatherv(local_C.data(), local_rows * N, MPI_DOUBLE,
        C.data(), send_counts.data(), displs.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD);

    // Процесс 0 выводит время и сохраняет результат
    if (rank == 0) {
        double elapsed = end_time - start_time;
        cout << "Calculation time: " << elapsed << " seconds" << endl;

        // Сохраняем результат в файл
        writeMatrix("mpi_result_" + to_string(N) + "_" + to_string(size) + ".txt", C, N);

        // Записываем время в общий файл
        ofstream timeFile("mpi_times.txt", ios::app);
        timeFile << N << " " << size << " " << elapsed << " 0" << endl;
        timeFile.close();
    }

    MPI_Finalize();
    return 0;
}