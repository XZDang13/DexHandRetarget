from pymodbus.client import ModbusSerialClient
from time import sleep
import numpy as np

class RealInspireHand:
    def __init__(self):
        print("Connecting to Inspire Hand")
        self.client = ModbusSerialClient(
        method="rtu",
        port="/dev/ttyUSB0",
        baudrate=115200,
        timeout=0.5,
        )
        self.client.connect()

        self.min_position = 0
        self.max_position = 2000   
        
        self.open_pose = 0
        self.closed_pose = 2000

        self.MAX_FINGER_POS = [1797, 1801, 1828, 1830, 1560, 1780]

        self.pinky_pos_addr = 0x05C2

    def send(self, ctrl):
        thumb_yaw = np.clip(self.MAX_FINGER_POS[5]*ctrl[5], 0, self.MAX_FINGER_POS[5])
        thumb_pitch = np.clip(self.MAX_FINGER_POS[4]*ctrl[4], 0, self.MAX_FINGER_POS[4])
        index = np.clip(self.MAX_FINGER_POS[3]*ctrl[3], 0, self.MAX_FINGER_POS[3])
        middle = np.clip(self.MAX_FINGER_POS[2]*ctrl[2], 0, self.MAX_FINGER_POS[2])
        ring = np.clip(self.MAX_FINGER_POS[1]*ctrl[1], 0, self.MAX_FINGER_POS[1])
        pinky = np.clip(self.MAX_FINGER_POS[0]*ctrl[0], 0, self.MAX_FINGER_POS[0])

        print(f"Sending to Inspire Hand: thumb_yaw={thumb_yaw}, thumb_pitch={thumb_pitch}, index={index}, middle={middle}, ring={ring}, pinky={pinky}")

        self.client.write_registers(self.pinky_pos_addr, [int(thumb_yaw), int(thumb_pitch), int(index), int(middle), int(ring), int(pinky)],slave=1)

        print(ctrl)