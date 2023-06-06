# Introduccion a las IA y Machine Learning


Si quiere pasar de ser programador a especialista en Al, éste es el lugar ideal para empezar. Basado en los cursos de Al de Laurence Moroney, de gran éxito.
de Laurence Moroney, este libro introductorio ofrece un enfoque práctico y
confianza mientras aprende los temas clave. Todo lo que necesita es experiencia
con Python y su notación para el procesamiento de datos y matrices.
Aprenderá a implementar los escenarios más comunes del aprendizaje automático, incluyendo visión por ordenador, 
procesamiento del lenguaje natural (NLP), y el modelado de secuencias para la web, móvil, nube, y tiempos de ejecución integrados. La mayoría de los libros sobre aprendizaje automático
comienzan con una cantidad desalentadora de matemáticas avanzadas. Esta guía ofrece
lecciones prácticas que le permiten trabajar directamente con el código.
#
## Introducción a TensorFlow
Cuando se trata de crear inteligencia artificial (IA), el aprendizaje automático (ML) y el
aprendizaje profundo son un gran lugar para comenzar. Sin embargo, al empezar, es fácil sentirse
abrumado por las opciones y toda la nueva terminología. Este libro pretende
tificar las cosas para los programadores, llevándoles a través de la escritura de código para implementar conceptos
de aprendizaje automático y aprendizaje profundo; y la construcción de modelos que se comportan más como un
humano, con escenarios como la visión por ordenador, el procesamiento del lenguaje natural (PLN) y otros,
etc. Así, se convierten en una forma de inteligencia sintetizada, o artificial.
Pero cuando nos referimos al aprendizaje automático, ¿qué es en realidad este fenómeno? Echemos un
y considerémoslo desde la perspectiva de un programador antes de seguir adelante.
antes de continuar. Después de eso, este capítulo le mostrará cómo instalar las herramientas del oficio,
desde el propio TensorFlow hasta los entornos donde puedes codificar y depurar tus modelos TensorFlow.
#

## ¿Qué es el aprendizaje automático?
Antes de entrar en los entresijos del aprendizaje automático, veamos cómo evolucionó a partir de la programación tradicional.
programación tradicional. Empezaremos examinando qué es la programación tradicional, y luego
casos en los que es limitada. A continuación veremos cómo ha evolucionado el ML para
casos, y como resultado ha abierto nuevas oportunidades para implementar nuevos escenarios,
desbloqueando muchos de los conceptos de la inteligencia artificial.
La programación tradicional consiste en escribir reglas, expresadas en un lenguaje de programación, que actúan sobre los datos y la información.
que actúan sobre los datos y nos dan respuestas. Esto se aplica a casi cualquier
algo se puede programar con código.
# 
## ¿Qué es TensorFlow?
TensorFlow es una plataforma de código abierto para crear y utilizar modelos de aprendizaje automático. Implementa muchos de los algoritmos y patrones comunes necesarios para el aprendizaje maqui
aprendizaje automático, ahorrándole la necesidad de aprender toda la matemática y la lógica subyacentes y permitiéndole centrarse en su trabajo.
te permite centrarte en tu escenario. Está dirigido a todos, desde aficionados a
desarrolladores profesionales, a investigadores que empujan los límites de la inteligencia artificial.
gencia artificial. Y lo que es más importante, también permite desplegar modelos en la web, en la nube, en dispositivos móviles y en sistemas integrados,
y sistemas embebidos. Cubriremos cada uno de estos escenarios en este libro.
#
## Instalacíon 
Por lo tanto, en su entorno Python, instalar TensorFlow es tan fácil como usar:
```sh
pip install tensorflow
```

Tenga en cuenta que a partir de la versión 2.1, esto instalará la versión GPU de TensorFlow por
por defecto. Anteriormente, utilizaba la versión para CPU. Por lo tanto, antes de instalar, asegúrese de que
tener una GPU compatible y todos los controladores necesarios para ello. Los detalles al respecto están disponibles
en TensorFlow.
Si no tienes la GPU o los controladores necesarios, puedes instalar la versión para CPU de
TensorFlow en cualquier Linux, PC o Mac con:

```sh
pip install tensorflow-cpu
```

Una vez en marcha, puedes probar tu versión de TensorFlow con el siguiente código
siguiente:

```python
import tensorflow as tf
print(tf.__version__)
```
#
## Introducción al aprendizaje automático
Como vimos anteriormente en el capítulo, el paradigma del aprendizaje automático es uno en el que usted
datos, esos datos están etiquetados, y queremos averiguar las reglas que hacen coincidir los
con las etiquetas. El escenario más simple posible para mostrar esto en código es el siguiente.
Considere estos dos conjuntos de números:

```python
X = –1, 0, 1, 2, 3, 4
Y = –3, –1, 1, 3, 5, 7
```

Existe una relación entre los valores X e Y (por ejemplo, si X es -1 entonces Y es -3,
si X es 3, Y es 5, etc.). ¿Te das cuenta?
Después de unos segundos, probablemente hayas visto que el patrón aquí es Y = 2X - 1. ¿Cómo
¿Cómo lo has conseguido? Cada persona lo resuelve de una forma, pero yo suelo oír la observación de que
observación de que X aumenta en 1 en su secuencia, e Y aumenta en 2; por lo tanto, Y = 2X
+/- algo. A continuación, miran cuando X = 0 y ven que Y = -1, por lo que calculan que
la respuesta podría ser Y = 2X - 1. A continuación, miran los otros valores y ven que esta
hipótesis "encaja", y la respuesta es Y = 2X - 1.
Esto es muy parecido al proceso de aprendizaje automático. Echemos un vistazo a algunos Tensor-
Flow que podrías escribir para que una red neuronal haga este cálculo por ti.
#
Aquí está el código completo, usando las APIs de TensorFlow Keras. No te preocupes si aún no tiene sentido.
sentido todavía; vamos a ir a través de él línea por línea:
```python
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
model.fit(xs, ys, epochs=500)
print(model.predict([10.0]))
```
Empecemos por la primera línea. Probablemente hayas oído hablar de las redes neuronales, y hayas
probablemente hayas visto diagramas que las explican usando capas de neuronas interconectadas, un
un poco como

[![N|Solid](https://i.imgur.com/SkWdN6z.png)](https://nodesource.com/products/nsolid)

Cuando veas una red neuronal como ésta, considera que cada uno de los "círculos" es una neurona
y cada una de las columnas de círculos como una capa. Así, en la Figura 1-18, hay tres
capas: la primera tiene cinco neuronas, la segunda tiene cuatro y la tercera tiene dos.

Si volvemos a nuestro código y miramos sólo la primera línea, veremos que estamos definiendo
la red neuronal más simple posible. Sólo hay una capa, y contiene sólo una
neurona:
```python
model = Sequential([Dense(units=1, input_shape=[1])])
```

Cuando usas TensorFlow, defines tus capas usando Sequential. Dentro del
Sequential, se especifica el aspecto de cada capa. Sólo tenemos una línea dentro de
nuestra Sequential, por lo que sólo tenemos una capa.

A continuación, define cómo se ve la capa utilizando la API keras.layers. Hay muchos
de diferentes tipos de capas, pero aquí estamos usando una capa Densa. "Dense" significa un conjunto de
neuronas completamente (o densamente) conectadas, que es lo que puedes ver en la Figura 1-18 donde
cada neurona está conectada a cada neurona de la capa siguiente. Es la forma más común
tipo de capa. Nuestra capa Densa tiene unidades=1 especificadas, así que tenemos sólo una capa densa con una neurona en toda nuestra capa.
con una neurona en toda nuestra red neuronal. Por último, cuando se especifica la primera
capa en una red neuronal (en este caso, es nuestra única capa), tiene que decirle cuál es la forma de los datos de entrada.

forma de los datos de entrada. En este caso nuestros datos de entrada es nuestro X, que es sólo un único
por lo que especificamos que esa es su forma.
La siguiente línea es donde realmente empieza la diversión. Veámoslo de nuevo:

```python
model.compile(optimizer='sgd', loss='mean_squared_error')
```