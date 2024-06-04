import time
from Transbot_Lib import Transbot

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

# Función para mover los motores
def move_motors():
    print("Moviendo motores por 10 segundos...")
    bot.set_car_motion(0.5, 0.0)  # Mover adelante a 50% de la velocidad máxima
    time.sleep(10)

    print("Deteniendo motores por 5 segundos...")
    bot.set_car_motion(0, 0)  # Detener los motores
    time.sleep(5)

    print("Moviendo motores por otros 10 segundos...")
    bot.set_car_motion(0.5, 0.0)  # Mover adelante a 50% de la velocidad máxima
    time.sleep(10)

    print("Deteniendo motores...")
    bot.set_car_motion(0, 0)  # Detener los motores

# Ejecutar la función de movimiento de motores
move_motors()
