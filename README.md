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

```sh
import tensorflow as tf
print(tf.__version__)
```