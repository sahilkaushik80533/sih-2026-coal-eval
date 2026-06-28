import time
import sys

def fl():
    # Intro messages
    intro_steps = [
        "Initializing environment...",
        "Loading configuration...",
        "Connecting to resources...",
        "Preparing workflow...",
        "Starting main process..."
    ]
    
    for step in intro_steps:
        print(step)
        time.sleep(2)  # short pause between intro lines
    
    print("\nProgress tracking started:\n")
    
    # Slow progress updates
    for i in range(1, 100):  # goes up to 9% (~45 minutes)
        sys.stdout.write(f"\rProgress: {i}%")
        sys.stdout.flush()
        time.sleep(30)  # 5 minutes per step
    print("\nProcess finished (simulation).")

try:
    fl()
except KeyboardInterrupt:
    print("\nProcess cancelled by user.")
