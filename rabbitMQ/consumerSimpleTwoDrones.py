#!/usr/bin/env python
import sys, time, threading, keyboard, atexit, pika, json, numpy as np
from codrone_edu.swarm import Swarm

# === Configuratie ===
NUM_DRONES = 2  # rood, blauw
kill = False
hover_mode = False  # hoveren na 'h'
start_hover = False  # pas bewegen na 'm'

# === Swarm setup ===
swarm = Swarm()
swarm.connect()  # detecteert automatisch de drones

# === Hoeken van het kleinere vierkant (wereldcoördinaten in meter) ===
corners = [
    (0.4, 0.2),   # hoek A
    (1.4, 0.27),  # hoek B
    (1.47, 1.15), # hoek C
    (0.25, 1.2)    # hoek D
]

# Beginposities per dronekleur (hoekindex)
drone_corners = {"red": 0, "blue": 2}

# Volgorde waarin ze bewegen
move_order = ["red", "blue"]

# Posities van drones uit RabbitMQ
drone_positions = {"red": None, "blue": None}


# === Veiligheids-handlers ===
def safe_land():
    """Land alle drones veilig."""
    global kill
    if kill:
        return
    kill = True
    try:
        print("\n>>> FAILSAFE: land alle drones veilig...")
        swarm.land()
    except Exception as e:
        print("⚠️ Fout bij landingscommando:", e)

atexit.register(safe_land)


# === Keyboard watcher ===
def watch_for_keys():
    global kill, hover_mode, start_hover
    while True:
        if keyboard.is_pressed("q"):
            print("\n>>> NOODSTOP geactiveerd! <<<")
            safe_land()
            break
        if keyboard.is_pressed("h") and not hover_mode:
            print("\n>>> Hoverpunten geactiveerd! <<<")
            hover_mode = True
        if keyboard.is_pressed("m") and hover_mode and not start_hover:
            print("\n>>> Start beweging geactiveerd! <<<")
            start_hover = True
        time.sleep(0.1)


# === Hulpfuncties ===
def is_corner_free(corner, all_positions, threshold=0.6):
    """Controleer of een hoek vrij is (geen drone binnen 0.6 m)."""
    for color, pos in all_positions.items():
        if pos is None:
            continue
        dist = np.linalg.norm(np.array(corner) - np.array(pos))
        if dist < threshold:
            return False
    return True


def move_drone_to(color, next_corner_idx):
    """Verplaats een drone naar de volgende hoek met move_forward/back/left/right."""
    try:
        idx = {"red": 0, "blue": 1}[color]
        current_idx = drone_corners[color]
        current_pos = corners[current_idx]
        target = corners[next_corner_idx]

        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]

        print(f"🟢 {color} beweegt van hoek {current_idx} naar hoek {next_corner_idx} → {target}")

        # Beweeg eerst X-richting
        if dx > 0:
            swarm[idx].move_right(dx, units='m', speed=0.3)
        elif dx < 0:
            swarm[idx].move_left(-dx, units='m', speed=0.3)

        # Daarna Y-richting
        if dy > 0:
            swarm[idx].move_forward(dy, units='m', speed=0.3)
        elif dy < 0:
            swarm[idx].move_backward(-dy, units='m', speed=0.3)

        # Update hoekindex
        drone_corners[color] = next_corner_idx

    except Exception as e:
        print(f"⚠️ Fout bij bewegen van {color}: {e}")
        safe_land()


# === RabbitMQ callback ===
def callback(ch, method, properties, body):
    try:
        msg = json.loads(body)
        color = msg.get("color")
        x = msg.get("x")
        y = msg.get("y")
        if color in drone_positions:
            drone_positions[color] = np.array([x, y])
    except Exception as e:
        print("⚠️ Fout bij RabbitMQ:", e)
        safe_land()


def rabbit_listener():
    """Lees continue droneposities uit RabbitMQ."""
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        channel = connection.channel()
        channel.queue_declare(queue='hello')
        channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)
        print("📡 Verbonden met RabbitMQ — posities worden bijgewerkt.")
        channel.start_consuming()
    except Exception as e:
        print("⚠️ RabbitMQ-verbinding mislukt:", e)
        safe_land()


# === Main ===
def main():
    global kill, hover_mode, start_hover
    try:
        threading.Thread(target=watch_for_keys, daemon=True).start()
        threading.Thread(target=rabbit_listener, daemon=True).start()

        # LED's instellen
        swarm.run_drone(0, "set_drone_LED", r=255, g=0, b=0, brightness=255)  # rood
        swarm.run_drone(1, "set_drone_LED", r=0, g=0, b=255, brightness=255)  # blauw

        print("Drones klaar. Druk op 'h' om te hoveren, 'm' om te starten met bewegen, 'q' om te stoppen.")

        # Wachten op hover ("h")
        while not hover_mode and not kill:
            time.sleep(0.1)

        if hover_mode and not kill:
            print("🚀 Drones stijgen op en blijven hoveren...")
            swarm.takeoff()
            swarm.hover(3)
            print("Drones hoveren. Druk op 'm' om te bewegen.")

        # Wachten op beweging ("m")
        while not start_hover and not kill:
            time.sleep(0.1)

        print("Drones starten met klokwijzerzinrotatie...")
        current_turn = 0
        last_print = 0

        while not kill:
            if time.time() - last_print > 0.5:
                print("📍 Posities:", {k: v.tolist() if v is not None else None for k, v in drone_positions.items()})
                last_print = time.time()

            color = move_order[current_turn]
            current_idx = drone_corners[color]
            next_idx = (current_idx + 1) % 4

            if is_corner_free(corners[next_idx], drone_positions):
                move_drone_to(color, next_idx)
            else:
                print(f"🔴 {color} wacht: hoek {next_idx} is bezet.")

            current_turn = (current_turn + 1) % len(move_order)
            time.sleep(3)

    except KeyboardInterrupt:
        print("\nCTRL+C gedetecteerd, landt drones...")
        safe_land()
    finally:
        try:
            swarm.disconnect()
            print("✅ Swarm losgekoppeld.")
        except:
            pass
        sys.exit(0)


if __name__ == "__main__":
    main()
