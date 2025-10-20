#!/usr/bin/env python
import pika, sys, threading, time, keyboard
from codrone_edu.swarm import Swarm
import json
import numpy as np

# === Swarm setup ===
swarm = Swarm()
swarm.connect()  # Detecteert automatisch alle beschikbare drones

# Config
CORRECTION_STEP = 0.05
TOLERANCE = 0.05
UPDATE_FREQUENCY = 10
WAYPOINT_SLEEP = 0.05

# Huidige positie en waypoint index
drone_positions = [None] * len(swarm.drones())
waypoint_idx = [0] * len(swarm.drones())

# Hardcoded waypoints per drone
drone_waypoints = [
    [(0.0,0.0), (0.5,0.0), (0.5,0.5), (0.0,0.5)],  # drone 0
    [(0.0,0.5), (0.5,0.5), (0.5,1.0), (0.0,1.0)],  # drone 1
    [(0.5,0.0), (1.0,0.0), (1.0,0.5), (0.5,0.5)]   # drone 2
]

# Kill flag en start flag
kill = False
start_correction = False

# === Noodstop en start key ===
def watch_for_keys():
    global kill, start_correction
    while True:
        if keyboard.is_pressed("q"):
            print("\n>>> NOODSTOP geactiveerd! <<<")
            kill = True
            try:
                swarm.land()
            except:
                pass
            break
        if keyboard.is_pressed("m"):
            print("\n>>> Start waypoint correctie! <<<")
            start_correction = True
        time.sleep(0.1)

# === Correctie functie ===
def move_drone_to(drone, current_pos, target_pos):
    dx = np.clip(target_pos[0] - current_pos[0], -CORRECTION_STEP, CORRECTION_STEP)
    dy = np.clip(target_pos[1] - current_pos[1], -CORRECTION_STEP, CORRECTION_STEP)
    if abs(dx) < TOLERANCE and abs(dy) < TOLERANCE:
        return True
    try:
        drone.move_distance(dx, dy, 0, WAYPOINT_SLEEP)
        current_pos[0] += dx
        current_pos[1] += dy
    except Exception as e:
        print("⚠️ Correctie fout:", e)
    return False

# === RabbitMQ callback ===
last_update_time = 0
def callback(ch, method, properties, body):
    global last_update_time, start_correction
    if kill:
        return
    if time.time() - last_update_time < 1 / UPDATE_FREQUENCY:
        return
    try:
        msg = json.loads(body)
        idx = msg.get("id", None)
        x = msg.get("x", 0)
        y = msg.get("y", 0)
        if idx is not None and idx < len(swarm.drones()):
            # Update huidige positie
            if drone_positions[idx] is None:
                drone_positions[idx] = np.array([x, y])
            else:
                drone_positions[idx][0] = x
                drone_positions[idx][1] = y

            # Hover correctie als start nog niet gedrukt
            if not start_correction:
                move_drone_to(swarm.drones()[idx], drone_positions[idx], drone_positions[idx])
            else:
                # Corrigeer richting waypoints
                wp_idx = waypoint_idx[idx]
                waypoints = drone_waypoints[idx]
                target = np.array(waypoints[wp_idx])
                reached = move_drone_to(swarm.drones()[idx], drone_positions[idx], target)
                if reached:
                    waypoint_idx[idx] = (wp_idx + 1) % len(waypoints)

            print(f"[Drone {idx}] pos=({x:.2f},{y:.2f}), waypoint_idx={waypoint_idx[idx]}")
    except Exception as e:
        print("⚠️ Fout bij verwerken bericht:", e)
    last_update_time = time.time()

# === Main loop ===
def main():
    global kill

    # Start key watcher thread
    watcher = threading.Thread(target=watch_for_keys, daemon=True)
    watcher.start()

    # LED kleuren als voorbeeld
    swarm.run_drone(0, "set_drone_LED", r=255, g=0, b=0, brightness=255)  # drone 0 rood
    swarm.run_drone(1, "set_drone_LED", r=0, g=255, b=0, brightness=255)  # drone 1 groen
    if len(swarm.drones()) > 2:
        swarm.run_drone(2, "set_drone_LED", r=0, g=0, b=255, brightness=255)  # drone 2 blauw

    # Alle drones opstijgen en hoveren
    swarm.takeoff()
    swarm.hover(2)

    # RabbitMQ connectie
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='hello')
    channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)

    print(" [*] Swarm actief. Druk 'm' voor waypoints, 'q' voor noodstop.")

    try:
        while not kill:
            connection.process_data_events()
            time.sleep(0.01)
    finally:
        print("\n>>> Failsafe actief — swarm landt nu veilig...")
        try:
            swarm.land()
            print("🛬 Alle drones zijn veilig geland.")
        except Exception as e:
            print("⚠️ Swarm landingsfout:", e)
        try:
            connection.close()
        except:
            pass
        sys.exit(0)

if __name__ == "__main__":
    main()
