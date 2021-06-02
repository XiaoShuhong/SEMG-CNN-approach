# Importing Libraries
from numpy.core.numeric import extend_all
import serial
import time
import threading
import numpy as np
import scipy as sp
import xgboost as xgb
from EMG_dsp_classify import predict
from collections import Counter

# Global variables
num_channels = 6                # 6 channels
step_size = 0.1                 # 0.1 second step
window_size = 5                 # 5 seconds window
samplerate = 500                # 500 Hz sampling
stopFlag = False                # Thread control
lock = threading.Lock()         # Thread sync
# Window buffer between read and write thread
window = np.zeros((num_channels, int(samplerate * window_size)))


class readThread (threading.Thread):
    def __init__(self, port, rate, timeout):
        threading.Thread.__init__(self, daemon=True)

        # Class constants
        global num_channels, step_size, window_size, samplerate
        self.num_channels = num_channels
        self.len_buffer = 60 * samplerate   # 60 second buffer
        self.step_size = step_size
        self.window_size = window_size
        self.samplerate = samplerate
        self.len_step = int(self.step_size * self.samplerate)
        self.len_stepbuf = 3 * self.len_step

        # Class variables
        self.scan = 0
        self.buffer = np.zeros((self.num_channels, self.len_buffer))
        self.stepbuf = np.zeros((self.num_channels, self.len_stepbuf))

        # Open arduino connection
        try:
            self.arduino = serial.Serial(
                port=port, baudrate=rate, timeout=timeout)
        except serial.serialutil.SerialException:
            print("Fail to open Serial port", port)

        # Ignore disturbance of first two seconds
        self.read_arduino()
        time.sleep(3)
        print("Starting read thread")

    def read_arduino(self):
        line = self.arduino.readline()
        try:
            s = line.decode().strip()
        except UnicodeDecodeError:
            # print("Decode error: ", line)
            return None
        lst = s.split(',')
        if len(lst) != self.num_channels + 1:
            # print("Invalid data: ", lst)
            return None
        for i in range(self.num_channels):
            if lst[i] == '':
                # print("Empty data: ", lst)
                return None
        return lst[:self.num_channels + 1]

    def read_data(self):
        global lock, window
        window_len = self.samplerate * self.window_size

        # Clear buffer if limit is reached
        if self.scan + self.len_step >= self.len_buffer:
            for i in range(self.num_channels):
                self.buffer[i, 0: window_len] = self.buffer[i,
                                                            self.scan - window_len: self.scan]
            self.scan = window_len

        # Read 100ms data
        start = time.time()

        # Read data into step buffer
        n_data = 0
        while time.time() - start < self.step_size:
            data = self.read_arduino()
            if data:
                for i in range(self.num_channels):
                    self.stepbuf[i, n_data] = int(data[i])
                n_data += 1
            if n_data == self.len_stepbuf:
                print("Samplerate too high (> 1500), discarding this sample.", sep='')
                lock.acquire()
                window[:, :] = 0
                lock.release()
                return

        if n_data == 0:
            print("Samplerate too low(", n_data,
                  "), discarding this sample.", sep='')
            lock.acquire()
            window[:, :] = 0
            lock.release()
            return

        # print("samples: ", n_data)
        # Interpolate data into samplerate
        # Put the new samples into buffer
        x = np.linspace(0, self.len_step, n_data)
        y_interp = np.zeros((self.num_channels, self.len_step))
        for i in range(self.num_channels):
            x_interp = np.linspace(0, self.len_step - 1, self.len_step)
            f = sp.interpolate.interp1d(
                x, self.stepbuf[i, : n_data], kind='nearest')
            y_interp[i, :] = f(x_interp)
            self.buffer[i, self.scan: self.scan +
                        self.len_step] = y_interp[i, :]
        self.scan += self.len_step
        # print("before interp:", self.stepbuf[1, : n_data])
        # print("after interp:", y_interp[1, :])
        # Pass the window to the window buffer
        if self.scan < window_len:
            return
        lock.acquire()
        for i in range(self.num_channels):
            window[i, :] = self.buffer[i, self.scan - window_len: self.scan]
        lock.release()

    def run(self):
        while True:
            self.read_data()
            global stopFlag
            if stopFlag:
                print("Stopping read thread")
                break

num_pred = 10

class writeThread (threading.Thread):
    def __init__(self, port, rate, timeout):
        threading.Thread.__init__(self, daemon=True)

        # Class constants
        global num_channels, step_size, window_size, samplerate, num_pred
        self.fs = samplerate
        self.t_win = window_size
        self.t_step = step_size
        self.t_win_decision = 0.5
        self.num_pred = num_pred

        # Class variables
        self.predictions = np.zeros((self.num_pred, 1))
        self.idx = 0

        # Connect to arduino
        try:
            self.arduino = serial.Serial(
                port=port, baudrate=rate, timeout=timeout)
        except serial.serialutil.SerialException:
            print("Fail to open Serial port", port)

        # Import the pre-trained model
        self.bst = xgb.Booster(model_file="model_xgboost.json")
        print("Starting write thread")

    def write_arduino(self, x):
        self.arduino.write(bytes(x, 'utf-8'))

    def write_data(self):
        global lock, window
        # Read the window data
        lock.acquire()
        data = window
        lock.release()

        # Do classification on the window data
        pred = predict(self.fs, self.t_win_decision, data.T, self.bst)
        self.predictions[self.idx] = int(pred[0])
        self.idx += 1

        # Make a decision on every 10 predicts
        if self.idx == self.num_pred:
            # Get mode of last few predicts
            counts = {0: 0, 1: 0, 2: 0}
            for p in self.predictions:
                counts[p[0]] += 1

            if counts[0] > counts[1] and counts[0] > counts[2]:
                action = 0
            elif counts[1] > counts[2]:
                action = 1
            else:
                action = 2

            print("Last", self.num_pred, "predictions:", self.predictions.T, "Action: ", action)
            self.write_arduino(str(action))
            self.idx = 0

    def run(self):
        global stopFlag
        while True:
            self.write_data()
            time.sleep(1. / self.num_pred)
            global stopFlag
            if stopFlag:
                print("Stopping write thread")
                break


thread1 = writeThread('COM4', 9600, .5)
thread2 = readThread('COM3', 115200, .1)

thread1.start()
thread2.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    stopFlag = True

thread2.join()
thread1.join()
