import os
import time
import subprocess
import psutil
import json

TRAINING_SCRIPT = "snake_pixel_nohup.py"
LOG_FILE = "nohup.out"  
COMMAND = "nohup python -u snake_pixel_nohup.py > nohup.out 2>&1 &"

def check_if_running():
    for proc in psutil.process_iter(attrs=['cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and TRAINING_SCRIPT in ' '.join(cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False

def restart_script():
    print(f"Restarting {TRAINING_SCRIPT}...")
    os.system(COMMAND)
    return
    with open(LOG_FILE, "a") as log:
        subprocess.Popen(
            ["nohup", "python", TRAINING_SCRIPT],
            stdout=log,
            stderr=log,
            preexec_fn=os.setpgrp
        )

if __name__ == "__main__":
    os.system(COMMAND)
    print(f"Monitoring {TRAINING_SCRIPT}...")
    while True:
        if not check_if_running():
            print(f"{TRAINING_SCRIPT} is not running. Restarting...")
            restart_script()
        else:
            print('\n\n')
            print(f"{TRAINING_SCRIPT} is running.")
            os.system("tail nohup.out")
        
        time.sleep(120)
