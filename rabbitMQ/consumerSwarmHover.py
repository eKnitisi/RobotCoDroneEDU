#!/usr/bin/env python
import pika, sys, threading, time, keyboard, atexit
from codrone_edu.swarm import Swarm
import json
import numpy as np

# Gebruik enkel 2 drones (0 = rood, 1 = blauw)
NUM_DRONES = 2

# === Configuratie ===
CORRECTION_TOLERANCE = 0.05
CORRECTION_POWER = 10
CORRECTION_TIME = 0.15
UPDATE_FREQUENCY = 10
last_command_time = [0] * NUM_DRONES
COMMAND_INTERVAL = 0.15  # minimum tijd tussen commando's per drone in seconden

# === Swarm setup ===
swarm = Swarm()
swarm.connect()  # detecteert automatisch de drones



# === Variabelen ===
drone_positions = [None] * NUM_DRONES
hover_points = [None] * NUM_DRONES
kill = False
set_hover = False   
start_hover = False 


# === Veiligheids-handlers ===
def safe_land():
    """Land alle drones veilig, maar disconnect niet hier."""
    try:
        print("\n>>> FAILSAFE: land alle drones veilig...")
        swarm.land()
    except Exception as e:
        print("⚠️ Fout bij landingscommando:", e)

atexit.register(safe_land)

def global_exception_hook(exctype, value, tb):
    print("\n⚠️ Onverwachte fout:", value)
    safe_land()
    sys.__excepthook__(exctype, value, tb)
sys.excepthook = global_exception_hook

# === Keyboard watcher ===
def watch_for_keys():
    global kill, start_hover, set_hover
    while True:
        if keyboard.is_pressed("q"):
            print("\n>>> NOODSTOP geactiveerd! <<<")
            kill = True
            safe_land()
            break
        if keyboard.is_pressed("h"):
            print("\n>>> Hoverpunten worden nu ingesteld! <<<")
            set_hover = True

        if keyboard.is_pressed("m"):
            print("\n>>> Start hover geactiveerd! <<<")
            start_hover = True
        time.sleep(0.1)

def correct_position(drone_idx, current_pos, hover_pos):
    """Stuur één kleine stap richting hover_pos, met rate limiting."""
    if hover_pos is None or current_pos is None:
        return

    dx = hover_pos[0] - current_pos[0]
    dy = hover_pos[1] - current_pos[1]

    if abs(dx) < CORRECTION_TOLERANCE and abs(dy) < CORRECTION_TOLERANCE:
        return


# Werkt niet => te veel commandos tegelijkertijd => drone connectie software crasht

    # Check rate limiter
    global last_command_time
    now = time.time()
    if now - last_command_time[drone_idx] < COMMAND_INTERVAL:
        return  # te snel, skip deze stap

    try:
        # X-correctie
        if dx > CORRECTION_TOLERANCE:
            swarm[drone_idx].go("right", CORRECTION_POWER, CORRECTION_TIME)
        elif dx < -CORRECTION_TOLERANCE:
            swarm[drone_idx].go("left", CORRECTION_POWER, CORRECTION_TIME)

        # Y-correctie
        if dy > CORRECTION_TOLERANCE:
            swarm[drone_idx].go("forward", CORRECTION_POWER, CORRECTION_TIME)
        elif dy < -CORRECTION_TOLERANCE:
            swarm[drone_idx].go("backward", CORRECTION_POWER, CORRECTION_TIME)

        # update timestamp
        last_command_time[drone_idx] = now

    except Exception as e:
        print(f"⚠️ Correctie fout voor drone {drone_idx}: {e}")
        safe_land()



# === RabbitMQ callback ===
last_update_time = 0
def callback(ch, method, properties, body):
    global last_update_time, start_hover, set_hover  
    if kill:
        return
    if time.time() - last_update_time < 1 / UPDATE_FREQUENCY:
        return

    try:
        msg = json.loads(body)
        color = msg.get("color", None)
        x = msg.get("x", 0)
        y = msg.get("y", 0)

        color_map = {"red": 0, "blue": 1}
        idx = color_map.get(color, None)
        if idx is None or idx >= NUM_DRONES:
            return

        if drone_positions[idx] is None:
            drone_positions[idx] = np.array([x, y])
        else:
            drone_positions[idx][0] = x
            drone_positions[idx][1] = y

        if set_hover and hover_points[idx] is None and drone_positions[idx] is not None:
            hover_points[idx] = drone_positions[idx].copy()
            print(f"hoverpoint {color}: x={hover_points[idx][0]:.2f}, y={hover_points[idx][1]:.2f}")

        if start_hover and hover_points[idx] is not None:   # ✅ enkel corrigeren na 'm'
            correct_position(idx, drone_positions[idx], hover_points[idx])

        print(f"[Drone {idx}] ({color}) pos=({x:.2f},{y:.2f}), hover={hover_points[idx]}")

    except Exception as e:
        print("⚠️ Fout bij verwerken bericht:", e)
        safe_land()

    last_update_time = time.time()

# === Main loop ===
def main():
    global kill, start_hover
    try:
        watcher = threading.Thread(target=watch_for_keys, daemon=True)
        watcher.start()

        # LED instellen
        swarm.run_drone(0, "set_drone_LED", r=255, g=0, b=0, brightness=255)  # rood
        swarm.run_drone(1, "set_drone_LED", r=0, g=0, b=255, brightness=255)  # blauw

        # === Main loop ===
        print("Drones klaar. Druk op 'h' om hoverpunten vast te leggen, daarna op 'm' om te starten met hover...")

        while not set_hover and not kill:
            time.sleep(0.1)

        print("✅ Hoverpunten ingesteld. Wacht nu op 'm' om te starten...")

        while not start_hover and not kill:
            time.sleep(0.1)

        if start_hover:
            print("Drones stijgen op en beginnen correctie.")
            swarm.takeoff()
            swarm.hover(2)

        # RabbitMQ connectie
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        channel = connection.channel()
        channel.queue_declare(queue='hello')
        channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)

        print(" [*] Swarm actief — druk 'q' om te landen en stoppen.")

        while not kill:
            connection.process_data_events()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nCTRL+C gedetecteerd, landt drones...")
        safe_land()
    except Exception as e:
        print("⚠️ Onverwachte fout in main:", e)
        safe_land()
    finally:
        # disconnect altijd maar pas hier
        try:
            swarm.disconnect()
            print("✅ Swarm losgekoppeld.")
        except:
            pass
        sys.exit(0)

if __name__ == "__main__":
    main()
