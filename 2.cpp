#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

int main() 
{
    ifstream dataset("dollar.csv");

    string line;
    vector <double> t;
    vector <double> x;

    vector <double> av_t(11);
    vector <double> av_xt(6);
    int N = 0;

    // Считываем данные 
    while (getline(dataset, line))
    {
        stringstream ss(line);
        string cell;
        vector<string> row;

        // Считываем значения строки
        while (getline(ss, cell, ';')) 
        {
            row.push_back(cell);
        }

        // Вычисляем сумму значений первого и второго столбца
        if (!row.empty()) 
        {
            t.push_back(stod(row[0]));
            if (row.size() > 1)
            {
                x.push_back(stod(row[1])); // Преобразуем строку в double
            }
        }
        N++;
    }
    dataset.close();

    for (int i = 0; i < 11; i++)
    {
        for (int j = 0; j < t.size(); j++)
        {
            double current_t = pow(t[j], i);
            av_t[i] += current_t / N;

            if (i < 6)
            {
                av_xt[i] += current_t * x[j] / N;
            }
        }
    }

    // ПЕРВЫЙ ЭТАП
    Eigen::MatrixXd mat1(6, 6); // Создаем матрицы

    // Заполняем матрицу значениями из вектора со сдвигом, начиная с последнего значения
    int begin = 0;

    for (auto i = mat1.rows() - 1; i >= 0; --i) 
    {
        int k = begin;
        for (auto j = mat1.cols() - 1; j >= 0; --j) 
        {
            mat1(i, j) = av_t[k];
            k++;
        }
        begin++;
    }
    cout << "First matrix: \n" << mat1 << endl;

    Eigen::MatrixXd mat2(6, 1);
    mat2 << av_xt[5],
        av_xt[4],
        av_xt[3],
        av_xt[2],
        av_xt[1],
        av_xt[0];
    cout << "Second matrix: \n" << mat2 << endl;

    // Перемножаем матрицы
    Eigen::MatrixXd result = mat1.inverse() * mat2;
    double A = result(0);
    double B = result(1);
    double C = result(2);
    double D = result(3);
    double E = result(4);
    double F = result(5);

    // Выводим результат
    cout << "Coeffs:\n A: " << A << "\n B: " << B << "\n C: " << C << "\n D: " << D << "\n E: " << E << "\n F: " << F << endl << endl;

    // ВТОРОЙ ЭТАП
    ofstream gradient("Error.csv");

    double min_Error = 0.0;
    double current_Error = 0.0;
    double lambda = 0.032;

    for (int i = 0; i < x.size(); i++)
    {
        min_Error += pow(A * pow(t[i], 5) + B * pow(t[i], 4) + C * pow(t[i], 3) + D * pow(t[i], 2) + E * t[i] + F - x[i], 2);
    }

    cout << "Min Error: " << min_Error << endl;

    cout << "Input your coeff A: ";
    cin >> A;
    cout << "Input your coeff B: ";
    cin >> B;
    cout << "Input your coeff C: ";
    cin >> C;
    cout << "Input your coeff D: ";
    cin >> D;
    cout << "Input your coeff E: ";
    cin >> E;
    cout << "Input your coeff F: ";
    cin >> F;

    for (int i = 0; i < x.size(); i++)
    {
        current_Error += pow(A * pow(t[i], 5) + B * pow(t[i], 4) + C * pow(t[i], 3) + D * pow(t[i], 2) + E * t[i] + F - x[i], 2);
    }

    cout << "Current Error: " << setprecision(7) << current_Error << endl;

    double previous_Error = 0.0;
    int k = 0;

    while (abs(current_Error - previous_Error) > 0.00000001)
    {
        previous_Error = current_Error;
        current_Error = 0.0;

        double dE_dA = 0.0;
        double dE_dB = 0.0;
        double dE_dC = 0.0;
        double dE_dD = 0.0;
        double dE_dE = 0.0;
        double dE_dF = 0.0;

        for (int i = 0; i < x.size(); i++)
        {
            dE_dA += 2 * (A * pow(t[i], 5) + B * pow(t[i], 4) + C * pow(t[i], 3) + D * pow(t[i], 2) + E * t[i] + F - x[i]) * pow(t[i], 5);
        }
        A -= lambda * dE_dA;

        for (int i = 0; i < x.size(); i++)
        {
            dE_dB += 2 * (A * pow(t[i], 5) + B * pow(t[i], 4) + C * pow(t[i], 3) + D * pow(t[i], 2) + E * t[i] + F - x[i]) * pow(t[i], 4);
        }
        B -= lambda * dE_dB;

        for (int i = 0; i < x.size(); i++)
        {
            dE_dC += 2 * (A * pow(t[i], 5) + B * pow(t[i], 4) + C * pow(t[i], 3) + D * pow(t[i], 2) + E * t[i] + F - x[i]) * pow(t[i], 3);
        }
        C -= lambda * dE_dC;

        for (int i = 0; i < x.size(); i++)
        {
            dE_dD += 2 * (A * pow(t[i], 5) + B * pow(t[i], 4) + C * pow(t[i], 3) + D * pow(t[i], 2) + E * t[i] + F - x[i]) * pow(t[i], 2);
        }
        D -= lambda * dE_dD;

        for (int i = 0; i < x.size(); i++)
        {
            dE_dE += 2 * (A * pow(t[i], 5) + B * pow(t[i], 4) + C * pow(t[i], 3) + D * pow(t[i], 2) + E * t[i] + F - x[i]) * t[i];
        }
        E -= lambda * dE_dE;

        for (int i = 0; i < x.size(); i++)
        {
            dE_dF += 2 * (A * pow(t[i], 5) + B * pow(t[i], 4) + C * pow(t[i], 3) + D * pow(t[i], 2) + E * t[i] + F - x[i]) * 1;
        }
        F -= lambda * dE_dF;


        for (int i = 0; i < x.size(); i++)
        {
            current_Error += pow(A * pow(t[i], 5) + B * pow(t[i], 4) + C * pow(t[i], 3) + D * pow(t[i], 2) + E * t[i] + F - x[i], 2);
        }


        if (k % 500 == 0)
        {
            cout << "#";
            gradient << setprecision(7) << current_Error << endl;
        }

        k++;

    }

    cout << endl;

    // Выводим результат
    cout << setprecision(7) << current_Error << endl;
    cout << "Coeffs:\n A: " << A << "\n B: " << B << "\n C: " << C << "\n D: " << D << "\n E: " << E << "\n F: " << F << endl;

    gradient.close();

    return 0;
}