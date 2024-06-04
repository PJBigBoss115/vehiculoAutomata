# vehiculoAutomata
ROS automata 

[Ir a pruebas de funcionamiento](#Pruebas-de-funcionamiento)

# Explicación del Código: Detección de Objetos y Control de Motores con TensorFlow y OpenCV
Este código utiliza un modelo preentrenado de TensorFlow y OpenCV para detectar objetos y controlar un robot Transbot basado en la detección de colores y objetos. A continuación se detalla la funcionalidad de cada sección del código.

## Importaciones
```python
import time
import cv2 as cv
import numpy as np
import tensorflow as tf
from Transbot_Lib import Transbot
```

Se importan las bibliotecas necesarias: time para manejar retrasos, cv2 para procesar imágenes, numpy para operaciones numéricas y tensorflow para el modelo de detección de objetos. También se importa la biblioteca Transbot_Lib para controlar el robot.

## Cargar el Modelo Preentrenado de TensorFlow
```python
model_dir = "ssd_mobilenet_v2/saved_model"
model = tf.saved_model.load(model_dir)
infer = model.signatures['serving_default']
```

Se carga el modelo preentrenado de TensorFlow desde el directorio especificado y se obtiene la firma de inferencia para realizar predicciones.

## Inicialización del Robot Transbot
```python
bot = Transbot()
bot.create_receive_threading()
bot.set_auto_report_state(True, forever=False)
bot.set_auto_report_state(False, forever=False)
bot.clear_auto_report_data()
```

Se crea un objeto Transbot llamado bot y se inicializa la recepción de datos, habilitando y deshabilitando el envío automático de datos y limpiando los datos en caché.

## Funciones de Control de Motores
### Mover a Velocidad Media
```python
def move_motors_medium_speed():
    bot.set_car_motion(0.5, 0.0)  # Mover adelante a 50% de la velocidad máxima
```

### Mover a Velocidad Reducida
```python
def move_motors_slow_speed():
    bot.set_car_motion(0.3, 0.0)  # Mover adelante a 30% de la velocidad máxima
```

### Ajustar Velocidad Basada en la Posición del Color
```python
def adjust_motor_speed_based_on_color_position(x, frame_width):
    center_threshold = frame_width // 3
    left_threshold = frame_width // 2 - center_threshold
    right_threshold = frame_width // 2 + center_threshold

    if x < left_threshold:
        bot.set_car_motion(0.2, 0.5)  # Girar a la izquierda
    elif x > right_threshold:
        bot.set_car_motion(0.5, 0.2)  # Girar a la derecha
    else:
        move_motors_medium_speed()
```

### Función Principal
```python
def main():
    def nothing(x):
        pass

    cv.namedWindow('Settings')

    cv.createTrackbar('H Lower', 'Settings', 51, 179, nothing)
    cv.createTrackbar('S Lower', 'Settings', 5, 255, nothing)
    cv.createTrackbar('V Lower', 'Settings', 30, 255, nothing)
    cv.createTrackbar('H Upper', 'Settings', 112, 179, nothing)
    cv.createTrackbar('S Upper', 'Settings', 142, 255, nothing)
    cv.createTrackbar('V Upper', 'Settings', 99, 255, nothing)

    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se puede abrir la cámara.")
        exit()

    red_detected = False
    yellow_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(input_tensor)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = infer(input_tensor)

        detection_boxes = detections['detection_boxes'].numpy()
        detection_classes = detections['detection_classes'].numpy().astype(np.int64)
        detection_scores = detections['detection_scores'].numpy()

        h_lower = cv.getTrackbarPos('H Lower', 'Settings')
        s_lower = cv.getTrackbarPos('S Lower', 'Settings')
        v_lower = cv.getTrackbarPos('V Lower', 'Settings')
        h_upper = cv.getTrackbarPos('H Upper', 'Settings')
        s_upper = cv.getTrackbarPos('S Upper', 'Settings')
        v_upper = cv.getTrackbarPos('V Upper', 'Settings')

        lower_bound = np.array([h_lower, s_lower, v_lower])
        upper_bound = np.array([h_upper, s_upper, v_upper])

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_bound, upper_bound)

        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)

        if cv.__version__.startswith('3.'):
            _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        else:
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        color_detected = False
        for contour in contours:
            if cv.contourArea(contour) > 500:
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                color_detected = True
                if not red_detected:
                    adjust_motor_speed_based_on_color_position(x + w // 2, frame.shape[1])

        for i in range(len(detection_scores)):
            if detection_scores[i] > 0.5:
                box = detection_boxes[i]
                y1, x1, y2, x2 = box
                x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                class_id = detection_classes[i]
                if class_id == 1:  # Suponiendo que 1 es el ID para rojo
                    red_detected = True
                    bot.set_car_motion(0, 0)
                elif class_id == 2:  # Suponiendo que 2 es el ID para verde
                    if red_detected:
                        red_detected = False
                elif class_id == 3:  # Suponiendo que 3 es el ID para amarillo
                    yellow_detected = True

        if red_detected:
            bot.set_car_motion(0, 0)
        elif yellow_detected:
            if color_detected:
                move_motors_slow_speed()
        elif color_detected:
            move_motors_medium_speed()
        else:
            bot.set_car_motion(0, 0)

        cv.imshow('Frame', frame)
        cv.imshow('Mask', mask)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    bot.set_car_motion(0, 0)

main()
```

## Descripción del Flujo Principal
1. Inicialización de la Ventana de Configuración:
      + Se crea una ventana llamada 'Settings' con controles deslizantes para ajustar los valores HSV.
2. Inicialización de la Captura de Video:
      + Se abre la cámara y se verifica si se ha abierto correctamente.
3. Bucle Principal:
      + Se captura cada frame de la cámara y se preprocesa para el modelo de TensorFlow.
      + Se realiza la detección de objetos utilizando el modelo de TensorFlow.
      + Se obtienen los valores de los controles deslizantes para definir el rango de color en HSV.
      + Se crea una máscara para el color de seguimiento y se encuentran los contornos en la máscara.
      + Si se detecta color, se ajusta la velocidad de los motores según la posición del color.
      + Se detectan colores rojo, verde y amarillo mediante TensorFlow y se ajustan las velocidades de los motores en consecuencia.
      + Se muestra el frame original y la máscara en ventanas de OpenCV.
4. Liberación de Recursos:
      + Se liberan la captura de video y se cierran todas las ventanas de OpenCV.

### Ejecución del Programa
La función main() se ejecuta al final para iniciar todo el proceso.

## Pruebas de funcionamiento

### Primera prueba
La primera prueba presenta un poco de ruido por lo cual el modelo detecta objetivos ajenos a la cinta.

![WhatsApp Image 2024-06-03 at 10 00 01 PM](https://github.com/PJBigBoss115/vehiculoAutomata/assets/65696918/2fd2413c-201b-4f8e-99da-a4c32e322393)

Video:

https://github.com/PJBigBoss115/vehiculoAutomata/assets/65696918/9ffef43f-8ef2-4a0b-b952-bbd4beaa1c07






