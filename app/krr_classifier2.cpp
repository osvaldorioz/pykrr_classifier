#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

//g++ -O2 -Wall -shared -std=c++20 -fPIC `python3.12 -m pybind11 --includes` krr_classifier2.cpp -o krr_classifier_mod`python3.12-config --extension-suffix` 

namespace py = pybind11;

class KRRClassifier {
public:
    KRRClassifier(double lambda) : lambda_(lambda) {}

    // Método para entrenar el modelo
    void fit(const std::vector<std::vector<double>> &X, const std::vector<double> &y) {
        int n_samples = X.size();
        
        if (n_samples == 0 || y.size() != n_samples) {
            throw std::invalid_argument("Dimensiones de X o y no válidas.");
        }

        // Calcula la matriz de kernel K
        std::vector<std::vector<double>> K = kernel(X, X);

        // Agrega lambda a la diagonal de K (ridge regularization)
        for (int i = 0; i < n_samples; ++i) {
            K[i][i] += lambda_;
        }

        // Resuelve K * alpha = y para alpha (utilizando inversión de matriz simple)
        alpha_ = solve_linear_system(K, y);
        X_train_ = X;
    }

    // Método para hacer predicciones
    std::vector<double> predict(const std::vector<std::vector<double>> &X) const {
        std::vector<std::vector<double>> K_test = kernel(X, X_train_);
        std::vector<double> predictions(K_test.size(), 0.0);

        for (size_t i = 0; i < K_test.size(); ++i) {
            for (size_t j = 0; j < alpha_.size(); ++j) {
                predictions[i] += K_test[i][j] * alpha_[j];
            }
        }

        return predictions;
    }

private:
    double lambda_;
    std::vector<double> alpha_;
    std::vector<std::vector<double>> X_train_;

    // Función de kernel (aquí se usa el kernel lineal)
    std::vector<std::vector<double>> kernel(const std::vector<std::vector<double>> &X1,
                                            const std::vector<std::vector<double>> &X2) const {
        size_t n1 = X1.size();
        size_t n2 = X2.size();
        std::vector<std::vector<double>> K(n1, std::vector<double>(n2, 0.0));

        for (size_t i = 0; i < n1; ++i) {
            for (size_t j = 0; j < n2; ++j) {
                K[i][j] = dot_product(X1[i], X2[j]);
            }
        }
        return K;
    }

    // Función para calcular el producto punto entre dos vectores
    double dot_product(const std::vector<double> &v1, const std::vector<double> &v2) const {
        double result = 0.0;
        for (size_t i = 0; i < v1.size(); ++i) {
            result += v1[i] * v2[i];
        }
        return result;
    }

    // Función para resolver el sistema K * alpha = y
    std::vector<double> solve_linear_system(const std::vector<std::vector<double>> &K,
                                            const std::vector<double> &y) const {
        size_t n = K.size();
        std::vector<std::vector<double>> K_inv = invert_matrix(K);
        std::vector<double> alpha(n, 0.0);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                alpha[i] += K_inv[i][j] * y[j];
            }
        }

        return alpha;
    }

    // Función para invertir una matriz cuadrada (asumiendo una inversa simple para el ejemplo)
    std::vector<std::vector<double>> invert_matrix(const std::vector<std::vector<double>> &K) const {
        // Esta implementación es solo para matrices pequeñas y no maneja todos los casos.
        size_t n = K.size();
        std::vector<std::vector<double>> K_inv(n, std::vector<double>(n, 0.0));

        for (size_t i = 0; i < n; ++i) {
            K_inv[i][i] = 1.0 / K[i][i];
        }

        return K_inv;
    }
};

// Enlace a Python con Pybind11
PYBIND11_MODULE(krr_classifier_mod, m) {
    py::class_<KRRClassifier>(m, "KRRClassifier")
        .def(py::init<double>())
        .def("fit", &KRRClassifier::fit)
        .def("predict", &KRRClassifier::predict);
}
