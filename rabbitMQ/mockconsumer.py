#!/usr/bin/env python
import threading, time, json, sys
import numpy as np
import pika

# === MockDrone voor simulatie ===
class MockDrone:
    def __init__(self, name):
        self.name = name
        self.position = np.array([0.0, 0.0])

    def pair(self):
        print(f"[{self.name}] pair() called")

    def takeoff(self):
        print(f"[{self.name}] takeoff() called")

    def land(self):
        print(f"[{self.name}] land() called")

    def move_distance(self, dx, dy, dz, duration):
        self.position += np.array([dx, dy])
        print(f"[{self.name}] moved to {self.position}")

    def hover(self, duration=0.5):
        print(f"[{self.name}] hover at {self.position}")

# === Config & drones ===
CORRECTION_STEP = 0.05
TOLERANCE = 0.05
UPDATE_FREQUENCY = 10

kill = False
drones = {
    "red": MockDrone("red"),
    "green": MockDrone("green"),
    "yellow": MockDrone("yellow")
}

drone_positions = {color: np.array([0.0, 0.0]) for color in drones}
last_update = {color: 0 for color in drones}

# === Waypoints voor driehoekpatroon (meters) ===
triangle_waypoints = {
    "red":   [[0.0,0.0], [0.5,0.5], [1.0,0.0], [0.0,0.0]],
    "green": [[0.0,0.5], [0.5,1.0], [1.0,0.5], [0.0,0.5]],
    "yellow": [[0.0,1.0], [0.5,1.5], [1.0,1.0], [0.0,1.0]]
}


waypoint_idx = {color: 0 for color in drones}

# === Noodstop ===
def watch_for_q():
    global kill
    import keyboard
    while True:
        if keyboard.is_pressed("q"):
            print("\n>>> NOODSTOP geactiveerd! <<<")
            kill = True
            for drone in drones.values():
                drone.land()
            break
        time.sleep(0.1)

# === Drone setup ===
def setup_drones():
    for drone in drones.values():
        drone.pair()
        drone.takeoff()
        drone.hover()

# === Correctie naar waypoint ===
def move_drone_to(drone, current_pos, target_pos):
    delta = target_pos - current_pos
    dx, dy = np.clip(delta, -CORRECTION_STEP, CORRECTION_STEP)
    distance = np.linalg.norm(delta)
    if distance < TOLERANCE:
        return True
    drone.move_distance(dx, dy, 0, 0.5)
    return False

# === Simulatie loop: vliegt automatisch langs driehoek ===
def simulation_loop():
    global kill
    while not kill:
        for color, drone in drones.items():
            idx = waypoint_idx[color]
            waypoints = np.array(triangle_waypoints[color])
            target = waypoints[idx]
            reached = move_drone_to(drone, drone_positions[color], target)
            drone_positions[color] = drone.position.copy()
            if reached:
                waypoint_idx[color] = (idx + 1) % len(waypoints)
        time.sleep(0.1)

# === RabbitMQ callback (werkt nog steeds) ===
def callback(ch, method, properties, body):
    try:
        msg = json.loads(body)
        color = msg["color"]
        x = msg["x"]
        y = msg["y"]
        if color in drones:
            drone_positions[color] = np.array([x, y])
            move_drone_to(drones[color], drones[color].position, drone_positions[color])
            last_update[color] = time.time()
            print(f"[{color}] corrected to ({x:.2f},{y:.2f})")
    except Exception as e:
        print("⚠️ Fout bij RabbitMQ bericht:", e)

# === Main loop ===
def main():
    global kill
    watcher = threading.Thread(target=watch_for_q, daemon=True)
    watcher.start()

    setup_drones()

    # Start simulatie
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()

    # RabbitMQ connectie
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='hello')
    channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)

    print(" [*] Simulatie gestart. Druk 'q' om te stoppen.")
    try:
        while not kill:
            connection.process_data_events()
            time.sleep(0.01)
    finally:
        for drone in drones.values():
            drone.land()
        connection.close()
        sys.exit(0)

if __name__ == "__main__":
    main()
