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

# Función para mover los motores de forma progresiva
def move_motors_gradually():
    print("Moviendo motores de forma progresiva por 10 segundos...")

    # Aumentar la velocidad de 0% a 100% en pasos del 10%
    for speed in range(1, 11):
        bot.set_car_motion(speed * 0.1, 0.0)  # Aumentar la velocidad progresivamente
        time.sleep(1)  # Mantener la velocidad actual durante 1 segundo

    print("Deteniendo motores por 5 segundos...")
    bot.set_car_motion(0, 0)  # Detener los motores
    time.sleep(5)

    print("Moviendo motores de forma progresiva por otros 10 segundos...")
    
    # Aumentar la velocidad de 0% a 100% en pasos del 10%
    for speed in range(1, 11):
        bot.set_car_motion(speed * 0.1, 0.0)  # Aumentar la velocidad progresivamente
        time.sleep(1)  # Mantener la velocidad actual durante 1 segundo

    print("Deteniendo motores...")
    bot.set_car_motion(0, 0)  # Detener los motores

# Ejecutar la función de movimiento de motores
move_motors_gradually()
