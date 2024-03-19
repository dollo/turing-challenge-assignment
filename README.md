# Turing Challenge Assignment for Data Scientist
Este reto está orientado a resolver, de una forma aproximada, un proyecto similar a algunos de los desarrollados en Turing Challenge desde el punto de vista del departamento de ciencia de datos. Con objeto de simplificar y reducir el tiempo de desarrollo no se tendrá en cuenta, aunque no se prohíbe el uso de Azure. Eso sí, no está permitido el uso de servicios cognitivos.



El reto estará compuesto de 3 apartados. Un apartado técnico, un apartado teórico y un apartado opcional.

## Código
Requisitos:
- **Python >= 3.10**
- Instalar dependencias de entorno virtual utilizando poetry o el fichero `requirements.txt`.

La app en gradio se puede ejectuar así desde el directorio raíz:

```bash
python turing_challenge_assignment/app.py
```

## Apartado 1
El objetivo del apartado técnico es valorar la capacidad de buscar una solución a un problema en un tiempo limitado y que ésta de la salida especificada. No se plantea evaluar que la solución sea optima en ninguna métrica, es decir, ni tiempo de ejecución, ni precisión de la inferencia.


Apartado 1: Crear un chatbot que tenga las siguientes funcionalidades:

    [x] Una interfaz, por ejemplo la interfaz de chatbot de Gradio.
    [x] Se han de ingestar varios documentos PDF largos para usarlos como base de conocimiento de un RAG. Se ha de usar una base de datos vectorial a elección.
    [x] Se ha de implementar una memoria dinámica que mantenga la conversación y que cuendo esta pase de X tokens se resuma de forma automática.
    [x] La implementación ha de estar basada en langchain.
    [ ] Si se detecta una pregunta que necesite de exactitud en la respuesta el modelo ha ser capaz de implementar y ejecutar código python.

## Apartado 2
El objetivo del apartado teórico es realizar un pequeño guion de los pasos necesarios a realizar para resolver el reto, así como detectar y comentar los posibles problemas que podrían aparecer y plantear contramedidas para minimizar los riesgos.

Apartado 2:  Dar respuesta a los siguientes puntos de forma teórica, sin necesidad de desarrollarlos, que guardan relación con las tecnologías utilizadas en el primer apartado:

### Diferencias entre 'completion' y 'chat' models:
Los modelos completion son útiles en tareas específcias, como hacer resúmenes, traducir idiomas, etc. Modelos tipo instruct como gpt-3.5-turbo-instruct están dentro de esta categoría.

Por otro lado, los modelos tipo chat están mejor preparados para conversación, es decir: conversaciones con varios turnos sistema-humano y refererirse a un histórico de la conversación.


### ¿Cómo forzar a que el chatbot responda 'si' o 'no'?¿Cómo parsear la salida para que siga un formato determinado?

- Opción 1: Prompting. Podemos darle en el prompt las instrucciones sobre las que queremos que procese determinado input y nos devuelva una salida binaria sí/no (por ejemplo, chain of thought y/o few-shot). Es conveniente ajustar la temperatura a 0 para evitar la mayor cantidad de ruido posible.
- Opción 2: Fine-tuning. Si tenemos una batería de ejemplos sobre la que sabemos cómo queremos que conteste nuestro LLM (es decir, en las que las contestaciones sean sí o no), podemos hacer run fine-tuning orientado a esa respuesta binaria. Con esta solución reducimos el número de tokens procesados en cada inferencia.
- Opción 3: Modificar logits. Ajustar las probabilidades de respuesta del modelo dando prioridad a los únicos tokens de salida que queremos que utilice: sí/no. Para ello hay que identificar los logits que le corresponden, darles mayor peso, y normalizar probabilidades (softmax).

### Ventajas e inconvenientes de RAG vs fine-tunning

**RAG (Retrieval Augmented Generation)** ofrece las siguientes ventajas y desventajas:
- **Ventajas**:
    - Alta adaptabilidad a dominio: podemos hacer un sistema muy
    - Control de las fuentes: sabemos cuáles son las fuentes de verdad sobre las que queremos generar respuestas.
    - Alta especificidad de las respuestas: datos muy concretos alojados en SQL o Excels pueden ser directamente accesibles y generar respuesta con ellos

- **Desventajas**:
    - Alta complejidad del sistema: cada dominio requiere de un ajuste fino del sistema, y la cadena de elementos que lo componen no es trivial:
        - Reformulador de queries: por ejemplo, subquery decomposition, step-back prompting.
        - Procesado de documentos: uso de embeddings adecuado (incluso hacer un fine-tuning), optimizar tamaño de fragmentos y solapamiento en base de datos de vectores.
        - Funciones de búsqueda en base de datos vectorial y otros recursos: contar con descripciones adecuadas de los recursos para poder acceder a ellas en varios niveles, no solamente por similitud coseno a fragmentos.
        - Generación de respuesta: influida por ténicas como el re-ranking, map-reduce
        - Prompting adecuado para cada una de las etapas/modelos utilizados
        - Manejo del contexto: problemas de tamaño en el contexto, needle-in-a-haystack...
    - Tiempo y coste: esta alta complejidad de elementos, muchos de ellos dependientes de llamdas a LLMs, hacen del sistema y costoso en términos de tiempo y dinero a nivel de inferencia total.

**Fine-tuning** tiene diversas ventajas y desventajas:
- **Ventajas**
    - Alta adaptabilidad a tarea: El fine-tuning nos permite adaptar el modelo a un tipo de interacción concreta, asociada quizás a alguna jerga (como la legal) o a algún dominio de conocimiento. Sin embargo, no debe confundirse con la introducción de conocimientos específicos.
    - Bajo control de las fuentes de verdad: por mucha información que haya en el fine-tuning, no podemos evitar que el modelo hile alucine, hilando hechos o conceptos que no tienen por qué ir juntos.
    - Baja especificidad: la generación de respuestas está más orientada en forma que en fondo.
- **Desventajas**:
    - Alto coste de entrenamiento (incluso de inferencia, aunque sea un modelo de OpenAI)
    - Necesita elaborar un dataset que refleje que permita el fine-tuning.

### ¿Cómo evaluar el desempaño de un bot de Q&A? ¿Cómo evaluar el desempeño de un RAG?

Un bot de Q&A va a tener que dar respuestas correctas a determinadas preguntas. Existen muchas métricas que permiten evaluar el acierto de un sistema a la hora de generar respuestas.

Respecto al **acierto del sistema**, tenemos métricas clásicas como accuracy, f1-score... Esta clasificación de respuesta requiere normalmente de validación manual o de llamadas a otro LLM que compare si la respuesta es correcta teniendo la respuesta correcta al lado para comparar.

También tenemos otras métricas que mejoran la **descripción de esa generación de respuesta**, añadiendo tintes cualitativos: perpejidad (perplexity), ROUGUE, similitud semántica basada en embeddings...

Si queremos evaluar el RAG directamente, aquí tenemos medidas que incluyen el contexto dentro de la evaluación, como faithfulness, answer relevancy, context precision y context recall. Hay muchas más, con herramientas (RAGAS) y artículos científicos (Compex QA Survey de Daull y cols. en 2023) dedicados específicamente a ello, refiriendo no solo sino benchmarks como NaturalQA, SQUAD, etc.

También tenemos otras medidas relacionadas con la **seguridad** (toxicidad, representación demográfica, sesgos... El paper de llama2 los describe en detalle), la **usabilidad** (tiempo de respuesta) y **satisfacción** subjetiva del usuario.


## Apartado 3
El objetivo del apartado opcional es simplemente una oportunidad para demostrar el conocimiento en otras tecnologías relevantes.


Apartado 3 (Opcional): Servicio local para detección de objetos. El objetivo es disponer de un servicio que tenga como entrada una imagen y que como salida proporcione un JSON con detecciones de coches y personas. Se han de cumplir los siguientes puntos:

    No hay necesidad de entrenar un modelo. Se pueden usar preentrenados.
    El servicio ha de estar conteinerizado. Es decir, una imagen docker que al arrancar exponga el servicio.
    La petición al servicio se puede hacer desde Postman o herramienta similar o desde código Python.
    La solución ha de estar implementada en Python.



Además, plantear cuales serían los pasos necesarios para entrenar un modelo de detección con categorías no existentes en los modelos preentrenados. Los puntos en los que centrar la explicación son:

    Pasos necesarios a seguir.
    Descripción de posibles problemas que puedan surgir y medidas para reducir el riesgo.
    Estimación de cantidad de datos necesarios así como de los resultados, métricas, esperadas.
    Enumeración y pequeña descripción (2-3 frases) de técnicas que se pueden utilizar para mejorar el desempeño, las métricas del modelo en tiempo de entrenamiento y las métricas del modelo en tiempo de inferencia.
