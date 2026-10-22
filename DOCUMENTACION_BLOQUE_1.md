# Explicación del bloque 1

En este bloque se prepara el corpus pequeño que se utilizará en los
experimentos de la Parte A. Primero se importan las librerías necesarias para
trabajar con NLTK, procesar texto, crear tablas y medir tiempos.

Después se carga la colección Gutenberg de NLTK. El código obtiene los
identificadores de los documentos y comprueba que estén disponibles los 18
libros solicitados. Si la colección todavía no está instalada, se descarga
automáticamente.

Para que todos los documentos puedan procesarse de la misma manera, se define
una configuración de normalización. El texto se convierte a minúsculas y se
extraen únicamente tokens formados por letras o números. La puntuación y otros
caracteres especiales se eliminan, mientras que los apóstrofes internos se
conservan. Por ejemplo, `Hello, world!` se convierte en `hello` y `world`.

Se guardan dos versiones de cada documento:

- El texto original, que permite conocer el tamaño real de cada archivo.
- La lista de tokens normalizados, que será utilizada por la búsqueda
  secuencial y el índice invertido.

Finalmente, se crea una tabla resumen con el identificador de cada documento,
la cantidad de caracteres originales y el número de tokens generados. También
se muestra el total de documentos y de tokens del corpus.

La idea principal es dejar todos los documentos preparados con la misma
representación antes de implementar las búsquedas. Así, la comparación entre
la búsqueda secuencial y el índice invertido se realiza sobre exactamente los
mismos datos.
