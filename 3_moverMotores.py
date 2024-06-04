import time
from Transbot_Lib import Transbot

# Crear un objeto Transbot llamado bot
bot = Transbot()

# Iniciar la recepciÃ³n de datos, solo puede iniciarse una vez, todas las funciones de lectura de datos se basan en este mÃ©todo
bot.create_receive_threading()

# Habilitar el envÃ­o automÃ¡tico de datos
enable = True
bot.set_auto_report_state(enable, forever=False)

# Deshabilitar el envÃ­o automÃ¡tico de datos
enable = False
bot.set_auto_report_state(enable, forever=False)

# Limpiar los datos en cachÃ© enviados automÃ¡ticamente por el MCU
bot.clear_auto_report_data()

# FunciÃ³n para mover los motores
def move_motors():
    print("Moviendo motores por 10 segundos...")
    bot.set_car_motion(0.5, 0.0)  # Mover adelante a 50% de la velocidad mÃ¡xima
    time.sleep(10)

    print("Deteniendo motores por 5 segundos...")
    bot.set_car_motion(0, 0)  # Detener los motores
    time.sleep(5)

    print("Moviendo motores por otros 10 segundos...")
    bot.set_car_motion(0.5, 0.0)  # Mover adelante a 50% de la velocidad mÃ¡xima
    time.sleep(10)

    print("Deteniendo motores...")
    bot.set_car_motion(0, 0)  # Detener los motores

# Ejecutar la funciÃ³n de movimiento de motores
move_motors()
