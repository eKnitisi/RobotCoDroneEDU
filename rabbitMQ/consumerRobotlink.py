#!/usr/bin/env python
import pika, sys, threading, time, keyboard
from codrone_edu.drone import Drone
import json
import numpy as np

# === Drones dictionary ===
drones = {
    "red": Drone(),
    "green": Drone(),
    "yellow": Drone()
}

# === CONFIG ===
CORRECTION_STEP = 0.05
TOLERANCE = 0.05
UPDATE_FREQUENCY = 10
WAYPOINT_SLEEP = 0.05  # korte pauze tussen correcties

# Huidige positie, timestamp en waypoint index
drone_positions = {color: None for color in drones}
last_update = {color: 0 for color in drones}
waypoint_idx = {color: 0 for color in drones}

# Hardcoded waypoints per drone (meters)
drone_waypoints = {
    "red": [(0.0,0.0), (0.5,0.0), (0.5,0.5), (0.0,0.5)],
    "green": [(0.0,0.5), (0.5,0.5), (0.5,1.0), (0.0,1.0)],
    "yellow": [(0.5,0.0), (1.0,0.0), (1.0,0.5), (0.5,0.5)]
}

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
            for drone in drones.values():
                try:
                    drone.land()
                except:
                    pass
            break
        if keyboard.is_pressed("m"):
            print("\n>>> Start waypoint correctie! <<<")
            start_correction = True
        time.sleep(0.1)

# === Drone setup ===
def setup_drones():
    for color, drone in drones.items():
        try:
            drone.pair()
            drone.takeoff()
            drone.hover()
        except Exception as e:
            print(f"⚠️ Fout bij {color} drone setup:", e)

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
        color = msg["color"]
        x = msg["x"]
        y = msg["y"]
        if color in drones:
            # Update huidige positie
            if drone_positions[color] is None:
                drone_positions[color] = np.array([x, y])
            else:
                drone_positions[color][0] = x
                drone_positions[color][1] = y

            last_update[color] = time.time()

            # Hover correctie als start nog niet gedrukt
            if not start_correction:
                move_drone_to(drones[color], drone_positions[color], drone_positions[color])
            else:
                # Corrigeer richting waypoints
                idx = waypoint_idx[color]
                waypoints = drone_waypoints[color]
                target = np.array(waypoints[idx])
                reached = move_drone_to(drones[color], drone_positions[color], target)
                if reached:
                    waypoint_idx[color] = (idx + 1) % len(waypoints)

            print(f"[{color}] pos=({x:.2f},{y:.2f}), waypoint_idx={waypoint_idx[color]}")
    except Exception as e:
        print("⚠️ Fout bij verwerken bericht:", e)
    last_update_time = time.time()

# === Main loop ===
def main():
    global kill
    watcher = threading.Thread(target=watch_for_keys, daemon=True)
    watcher.start()
    setup_drones()

    # RabbitMQ connectie
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='hello')
    channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)

    print(" [*] Drones zijn opgestegen en hoveren. Druk 'm' om waypoints te volgen, 'q' voor noodstop.")

    try:
        while not kill:
            connection.process_data_events()
            time.sleep(0.01)
    finally:
        for drone in drones.values():
            try:
                drone.land()
            except:
                pass
        connection.close()
        sys.exit(0)

if __name__ == "__main__":
    main()
