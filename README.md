# TT2

Este repositorio contiene los archivos de la interfaz de usuario, creada en python 3.9, así como los programas de Algoritmo Genético y Enjambre de Partículas.
Tambien contiene la versión de prueba del calculo de Hipervolumen. A continuación se describiran cada uno de los programas ya mencionados.

## GUI
Nombre del archivo: GUI_TT_MIOPTION.py <br>
La implementación de la GUI fue realizada con la biblioteca de PySimpleGUI, la cual fue seleccionada por su simplicidad de uso.
La interfaz grafica permite al usuario seleccionar el sistema a sintonizar, ingresar los parámetros dinámicos característicos de cada sistema, así como sleccionar el algoritmo metaheurístico deseado, posteriormente se deberan ingresar los parámetros necesarios de cada algoritmo, según sea el caso.
Al aceptar la ejeción del algoritmo seleccionado, se presentara la aproximación al frente de Pareto junto con una tabla que presenta el conjunto de soluciones inmejorables obtenidas por el algoritmo evolutivo.
Por último se puede seleccionar cualquier conjunto de ganancias y simular el comportamiento dinámico del péndulo seleccionado presionando el boton de simular, la simulación mostrara dos pestañas, una con la animación del comportamiento esperado y otra con las gráficas de posición, velocidad y señal de alimentación.

Librerias requeridas: <br>

* drawnow versión 0.72.5
* ipython versión 8.0.1
* matplotlib versión 3.4.3
* mplcursors versión 0.5.1
* numpy versión 1.21.2
* PySimpleGUI versión 4.56.0
* PySimpleGUIQt versión 0.35.0
* scipy versión 1.7.1
