import cv2 as cv
import rospy
from std_msgs.msg import Float64

"""
Este codigo inicia la camara ademas de inicializar los servos, se crea un nodo ros por lo
cual debe de estar configurado de manera correcta, en mi caso no funciono ya que no he
configurado el master pero ya configurando de manera correcta deberia funcionar sin problemas
"""

def move_servo(pan_pub, tilt_pub, pan_angle, tilt_angle):
    pan_pub.publish(pan_angle)
    tilt_pub.publish(tilt_angle)
    rospy.loginfo(f"Pan: {pan_angle}, Tilt: {tilt_angle}")

if __name__ == "__main__":
    rospy.init_node('servo_teleop', anonymous=True)

    pan_pub = rospy.Publisher('/servo/pan', Float64, queue_size=10)
    tilt_pub = rospy.Publisher('/servo/tilt', Float64, queue_size=10)

    frame = cv.VideoCapture(0)  # Inicializa la captura de video ID 0

    pan_angle = 0.0
    tilt_angle = 0.0
    step_size = 0.1  # Ajusta el tamaño del paso segun sea necesario

    while frame.isOpened() and not rospy.is_shutdown():
        ret, img = frame.read()  # Lee un frame de la camara
        if not ret:
            break  # Si no se puede leer el frame, rompe el bucle

        cv.imshow('frame', img)  # Muestra el frame en una ventana llamada 'frame'

        action = cv.waitKey(10) & 0xFF  # Espera 10 ms por una tecla y obtiene el codigo de la tecla presionada

        if action == ord('q') or action == 113:  # Si se presiona 'q' o 'Q', sale del bucle
            break
        elif action == 82:  # Flecha hacia arriba
            tilt_angle += step_size
        elif action == 84:  # Flecha hacia abajo
            tilt_angle -= step_size
        elif action == 81:  # Flecha hacia la izquierda
            pan_angle -= step_size
        elif action == 83:  # Flecha hacia la derecha
            pan_angle += step_size

        # Limita los angulos a un rango especifico (ajusta segun sea necesario)
        pan_angle = max(min(pan_angle, 1.57), -1.57)  # Limitar entre -90 y 90 grados
        tilt_angle = max(min(tilt_angle, 1.57), -1.57)  # Limitar entre -90 y 90 grados

        move_servo(pan_pub, tilt_pub, pan_angle, tilt_angle)  # Mueve los servos

    frame.release()  # Libera la camara
    cv.destroyAllWindows()  # Cierra todas las ventanas de OpenCV

