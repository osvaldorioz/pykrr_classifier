La regresión de cresta de núcleo (Kernel Ridge Regression, KRR) es un modelo de aprendizaje automático que utiliza una función de kernel para predecir una variable objetivo. Se trata de un método de regresión no lineal que puede manejar relaciones no lineales entre las variables de entrada y salida. 

KRR combina la regresión de cresta con el truco de kernel para aprender una función lineal en el espacio inducido por el kernel. Esto ayuda a obtener predicciones precisas al eliminar el sesgo y mejorar el rendimiento del modelo. 

El algoritmo de regresión de kernel requiere trabajar con una matriz N x N, donde N es el número de casos.

El programa es una implementación de Kernel Ridge Regression (KRR). La implementación hace 
uso de la librería estándar de C++ para realizar operaciones básicas de álgebra lineal.

Se implementan manualmente las funciones para multiplicar matrices, calcular el kernel y 
resolver el sistema lineal con la inversa de la matriz (con el enfoque de Ridge Regression).