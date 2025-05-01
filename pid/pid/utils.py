import os

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error):
        # Update I and D internally, then return PID output
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

def save_pid(Kp, Ki, Kd, path="best_pid.txt"):
    with open(path, "w") as f:
        f.write(f"{Kp},{Ki},{Kd}")

def load_pid(path="best_pid.txt"):
    if not os.path.exists(path):
        # Default PID gains
        return 0.1, 0.01, 0.05
    with open(path) as f:
        Kp, Ki, Kd = map(float, f.read().split(","))
    return Kp, Ki, Kd
