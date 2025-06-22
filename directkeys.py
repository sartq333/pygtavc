# reference:
# https://pythonprogramming.net/direct-input-game-python-plays-gta-v/
# http://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
# the above two links did not work for me as i'm on linux
# https://github.com/gauthsvenkat/pyKey/blob/master/pyKey/linux.py (this works for me)

import subprocess 
import time

W = "w"
A = "a"
S = "s"
D = "d"

def PressKey(key):
    subprocess.run(["xdotool", "keydown", key])

def ReleaseKey(key):
    subprocess.run(["xdotool", "keyup", key])

if __name__ == '__main__':
    PressKey(0x11)
    time.sleep(1)
    ReleaseKey(0x11)
    time.sleep(1)
