# Sistema de detección y seguimiento de personas

Realizado por: Álvaro González Rodríguez

En este repositorio se encuentra el código desarrollado en el proyecto final de máster: Sistema de detección y seguimiento de personas, del máster Big Data y Data Science de la Universidad Internacional de Valencia.

## Requisitos:

En este repositorio falta por añadir dos archivos de configuración. Estos son *mars-small128.pb* y *yolov4.h5*. Pueden ser adquiridos en los siguientes enlaces. Una vez descargados, añadirlos a la carpeta /model_data/models/

### Descargar mars-small128.pb:
https://drive.google.com/file/d/1fssmOGCt6A6QRefP17q_xNgjbQ-tooz8/view?usp=sharing

### Descargar pesos de YOLOv4:
https://drive.google.com/file/d/1MQ66SccSYpxndS9NIUYh-2yUEtOXcc6y/view?usp=sharing


## Comando para ejecutar la aplicación:

python app.py --input .\videos\input\4P-C1.mp4
python app.py --input .\videos\input\{video_a_su_eleccion}

Este comando devolverá una serie de vídeos con el seguimiento de las personas, un fichero de texto con información de las posiciones y un fichero .npy con los vectores de características.

python detections_from_serialized.py --input .\videos\input\4P-C2.mp4 --features .\videos\output\4P-C1\features.npy
python detections_from_serialized.py --input .\videos\input\{video_a_su_eleccion} --features .\videos\output\{video_elegido_en_ejecucion_app.py}\features.npy

Este comando muestra el vídeo de *--input* con el seguimiento de las personas que aparecían en el vídeo introducido en *--features*. 


## Reconocimientos
Esta aplicación ha sido creada gracias al increíble trabajo anterior de los siguientes repositorios:
  * https://github.com/samihormi/Multi-Camera-Person-Tracking-and-Re-Identification
  * https://github.com/KaiyangZhou/deep-person-reid
  * https://github.com/nwojke/deep_sort
