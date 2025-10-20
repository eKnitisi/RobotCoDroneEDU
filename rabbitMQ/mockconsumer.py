from codrone_edu.swarm import Swarm
import time

# Maak swarm-object
swarm = Swarm()
swarm.connect()  # detecteert automatisch alle beschikbare drones

try:
    # Zet kleuren: drone 0 rood, drone 1 groen
    swarm.run_drone(0, "set_drone_LED", r=255, g=0, b=0, brightness=128)
    swarm.run_drone(1, "set_drone_LED", r=0, g=0, b=255, brightness=255)

    # Alle drones opstijgen
    
    swarm.takeoff()
    time.sleep(2)

    # Hover 3 seconden
    swarm.hover(3)

    # Drone 1 (index 1) landt eerst
    print("Drone 1 landt")
    swarm[1].land()
    time.sleep(2)

    # Drone 0 (index 0) landt daarna
    print("Drone 0 landt")
    swarm[0].land()

finally:
    swarm.land()
    swarm.disconnect()
