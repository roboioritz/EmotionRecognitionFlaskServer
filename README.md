# EmotionRecognitionFlaskServer
A Flask Server with multiple emotion related stuff

Librerias a modificar tras la Pyinstalación del script:

Sustituir las librerias cv2, sklearn y scipy de la carpeta del ejecutable por las de la carpeta venv\lib

Copiar la libreria Librosa desde la carpeta venv\lib a la carpeta del ejecutable

Output del servidor: [Speech_emotion;Text_emotion;Face_emotion] (1 en caso de ser ira 0 en caso contrario)

Excepcion: NoText aparecerá en caso de que el audio no contenga texto

Excepcion: NoFace aparecerá en caso de que La API de reconocimiento facial no detecte ninguna cara

En caso de ambas excepciones aparecerá solo NoText
