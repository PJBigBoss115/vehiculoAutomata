import time
import cv2 as cv
import numpy as np
import tensorflow as tf
from Transbot_Lib import Transbot

# Cargar el modelo preentrenado de TensorFlow
model_dir = "ssd_mobilenet_v2/saved_model"
model = tf.saved_model.load(model_dir)

# Obtener la firma de inferencia
infer = model.signatures['serving_default']

# Crear un objeto Transbot llamado bot
bot = Transbot()

# Iniciar la recepción de datos, solo puede iniciarse una vez. Todas las funciones de lectura de datos se basan en este método.
bot.create_receive_threading()

# Habilitar el envío automático de datos
enable = True
bot.set_auto_report_state(enable, forever=False)

# Deshabilitar el envío automático de datos
enable = False
bot.set_auto_report_state(enable, forever=False)

# Limpiar los datos en caché enviados automáticamente por el MCU
bot.clear_auto_report_data()

# Función para mover los motores a una velocidad media
def move_motors_medium_speed():
    bot.set_car_motion(0.5, 0.0)  # Mover adelante a 50% de la velocidad máxima

# Función para mover los motores a una velocidad reducida
def move_motors_slow_speed():
    bot.set_car_motion(0.3, 0.0)  # Mover adelante a 30% de la velocidad máxima

# Función para ajustar la velocidad de los motores según la posición del color
def adjust_motor_speed_based_on_color_position(x, frame_width):
    center_threshold = frame_width // 3  # Umbral para considerar el centro
    left_threshold = frame_width // 2 - center_threshold
    right_threshold = frame_width // 2 + center_threshold

    if x < left_threshold:  # El color está a la izquierda
        bot.set_car_motion(0.2, 0.5)  # Reducir la velocidad del motor derecho, girar a la izquierda
    elif x > right_threshold:  # El color está a la derecha
        bot.set_car_motion(0.5, 0.2)  # Reducir la velocidad del motor izquierdo, girar a la derecha
    else:  # El color está en el centro
        move_motors_medium_speed()

# Función principal para detección de color y movimiento de motores
def main():
    def nothing(x):
        pass

    # Crear una ventana de configuración
    cv.namedWindow('Settings')

    # Crear controles deslizantes para ajustar los valores HSV
    cv.createTrackbar('H Lower', 'Settings', 51, 179, nothing)
    cv.createTrackbar('S Lower', 'Settings', 5, 255, nothing)
    cv.createTrackbar('V Lower', 'Settings', 30, 255, nothing)
    cv.createTrackbar('H Upper', 'Settings', 112, 179, nothing)
    cv.createTrackbar('S Upper', 'Settings', 142, 255, nothing)
    cv.createTrackbar('V Upper', 'Settings', 99, 255, nothing)

    # Inicializar la captura de video
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se puede abrir la cámara.")
        exit()

    # Inicializar flags para control de estados
    red_detected = False
    yellow_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocesar la imagen para el modelo de TensorFlow
        input_tensor = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(input_tensor)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Realizar la detección de objetos
        detections = infer(input_tensor)

        # Procesar las detecciones
        detection_boxes = detections['detection_boxes'].numpy()
        detection_classes = detections['detection_classes'].numpy().astype(np.int64)
        detection_scores = detections['detection_scores'].numpy()

        # Obtener los valores de los controles deslizantes
        h_lower = cv.getTrackbarPos('H Lower', 'Settings')
        s_lower = cv.getTrackbarPos('S Lower', 'Settings')
        v_lower = cv.getTrackbarPos('V Lower', 'Settings')
        h_upper = cv.getTrackbarPos('H Upper', 'Settings')
        s_upper = cv.getTrackbarPos('S Upper', 'Settings')
        v_upper = cv.getTrackbarPos('V Upper', 'Settings')

        # Definir el rango de color en HSV para el color de seguimiento
        lower_bound = np.array([h_lower, s_lower, v_lower])
        upper_bound = np.array([h_upper, s_upper, v_upper])

        # Crear una máscara para el color de seguimiento
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower_bound, upper_bound)

        # Realizar operaciones morfológicas para limpiar la máscara
        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)

        # Encontrar contornos en la máscara
        if cv.__version__.startswith('3.'):
            _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        else:
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Dibujar contornos en el frame original y mover motores si se detecta color
        color_detected = False
        for contour in contours:
            if cv.contourArea(contour) > 500:  # Filtrar pequeños contornos
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                color_detected = True
                if not red_detected:
                    adjust_motor_speed_based_on_color_position(x + w // 2, frame.shape[1])  # Ajustar velocidad basado en la posición del color

        # Detectar colores rojo, verde y amarillo mediante TensorFlow
        for i in range(len(detection_scores)):
            if detection_scores[i] > 0.5:
                box = detection_boxes[i]
                y1, x1, y2, x2 = box
                x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Obtener la clase detectada
                class_id = detection_classes[i]
                if class_id == 1:  # Suponiendo que 1 es el ID para rojo
                    red_detected = True
                    bot.set_car_motion(0, 0)  # Detener motores si se detecta rojo
                elif class_id == 2:  # Suponiendo que 2 es el ID para verde
                    if red_detected:
                        red_detected = False  # Reiniciar estado si se detecta verde
                elif class_id == 3:  # Suponiendo que 3 es el ID para amarillo
                    yellow_detected = True

        # Ajustar velocidades basadas en la detección de colores
        if red_detected:
            bot.set_car_motion(0, 0)  # Detener motores si se detecta rojo
        elif yellow_detected:
            if color_detected:
                move_motors_slow_speed()  # Reducir velocidad si se detecta amarillo
        elif color_detected:
            move_motors_medium_speed()  # Seguir el color a velocidad media
        else:
            bot.set_car_motion(0, 0)  # Detener motores si no se detecta color

        # Mostrar el frame original y la máscara
        cv.imshow('Frame', frame)
        cv.imshow('Mask', mask)

        # Salir del bucle si se presiona 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la captura de video y cerrar ventanas
    cap.release()
    cv.destroyAllWindows()
    bot.set_car_motion(0, 0)  # Asegurarse de detener los motores al final

# Ejecutar la función principal
main()
