#!/usr/bin/env python
import pika, sys, os
import threading
import time
import keyboard
from codrone_edu.drone import Drone

# RabbitMQ community docker image
# docker run -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:4-management


# === Drone setup ===
drones = {
    "red": Drone(),
    "green": Drone(),
    "yellow": Drone()
}

# Veiligheidsflag
kill = False

def watch_for_q():
    global kill
    while True:
        if keyboard.is_pressed("q"):
            print("\n>>> NOODSTOP geactiveerd! <<<")
            kill = True
            for color, drone in drones.items():
                try:
                    print(f"Drone {color} gaat veilig landen...")
                    drone.land()
                except Exception as e:
                    print(f"⚠️ Fout bij landen van {color}:", e)
            break
        time.sleep(0.1)

# Start noodstop watcher thread
watcher = threading.Thread(target=watch_for_q, daemon=True)
watcher.start()

# Pair en takeoff
for color, drone in drones.items():
    try:
        drone.pair()
        drone.takeoff()
    except Exception as e:
        print(f"⚠️ Fout bij initialisatie van {color}:", e)

# === RabbitMQ callback ===
def callback(ch, method, properties, body):
    global kill
    if kill:
        return  # negeer nieuwe berichten als noodstop geactiveerd is
    try:
        message = body.decode()
        print(f" [x] Received {message}")
        # TODO: hier kun je code toevoegen om drones aan te sturen op basis van de posities
    except Exception as e:
        print("⚠️ Fout bij verwerken bericht:", e)

# === Main consumer loop ===
def main():
    global kill
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='hello')

    channel.basic_consume(queue='hello', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. Druk "q" voor noodstop en exit.')
    try:
        while not kill:
            channel._process_data_events(time_limit=1)  # check voor berichten
    except KeyboardInterrupt:
        kill = True
    finally:
        print("\n=== Beëindigen: drones veilig landen en disconnect ===")
        for color, drone in drones.items():
            try:
                drone.land()
            except Exception as e:
                print(f"⚠️ Fout tijdens landen van {color}:", e)
        connection.close()
        sys.exit()

if __name__ == '__main__':
    main()
