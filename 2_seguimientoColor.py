import cv2 as cv
import numpy as np

def nothing(x):
    pass

# Crear una ventana de configuraciÃ³n
cv.namedWindow('Settings')

# Crear controles deslizantes para ajustar los valores HSV
cv.createTrackbar('H Lower', 'Settings', 0, 179, nothing)
cv.createTrackbar('S Lower', 'Settings', 0, 255, nothing)
cv.createTrackbar('V Lower', 'Settings', 0, 255, nothing)
cv.createTrackbar('H Upper', 'Settings', 179, 179, nothing)
cv.createTrackbar('S Upper', 'Settings', 255, 255, nothing)
cv.createTrackbar('V Upper', 'Settings', 255, 255, nothing)

# Inicializar la captura de video
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se puede abrir la cÃ¡mara.")
    exit()

# Obtener la version de OpenCV
(major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame a espacio de color HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Obtener los valores de los controles deslizantes
    h_lower = cv.getTrackbarPos('H Lower', 'Settings')
    s_lower = cv.getTrackbarPos('S Lower', 'Settings')
    v_lower = cv.getTrackbarPos('V Lower', 'Settings')
    h_upper = cv.getTrackbarPos('H Upper', 'Settings')
    s_upper = cv.getTrackbarPos('S Upper', 'Settings')
    v_upper = cv.getTrackbarPos('V Upper', 'Settings')

    # Definir el rango de color en HSV
    lower_bound = np.array([h_lower, s_lower, v_lower])
    upper_bound = np.array([h_upper, s_upper, v_upper])

    # Crear una mascara para el color
    mask = cv.inRange(hsv, lower_bound, upper_bound)

    # Realizar operaciones morfologicas para limpiar la mascara
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    # Encontrar contornos en la mascara
    if int(major_ver) < 4:
        # Para OpenCV 3.x
        _, contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    else:
        # Para OpenCV 4.x
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Dibujar contornos en el frame original
    for contour in contours:
        if cv.contourArea(contour) > 500:  # Filtrar pequeños contornos
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar el frame original y la mascara
    cv.imshow('Frame', frame)
    cv.imshow('Mask', mask)

    # Salir del bucle si se presiona 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar ventanas
cap.release()
cv.destroyAllWindows()
