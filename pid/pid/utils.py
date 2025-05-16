import os  # Filesystem operations for saving and loading PID parameters

class PIDController:
    """
    Simple PID controller to compute control actions based on error input.
    Maintains internal integral and derivative state between calls.
    """
    def __init__(self, Kp, Ki, Kd):
        # PID gain coefficients
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        # Integral accumulator starts at zero
        self.integral = 0.0
        # Previous error for derivative term calculation
        self.prev_error = 0.0

    def compute(self, error):
        """
        Compute the PID output for the given error.

        Steps:
        1. Accumulate the error into the integral term.
        2. Compute the derivative term as difference from previous error.
        3. Update previous error state.
        4. Return the weighted sum: P*error + I*integral + D*derivative.
        """
        # Update integral term
        self.integral += error
        # Compute derivative term
        derivative = error - self.prev_error
        # Store current error for next derivative computation
        self.prev_error = error
        # Return combined PID control output
        return (self.Kp * error
                + self.Ki * self.integral
                + self.Kd * derivative)


def save_pid(Kp, Ki, Kd, path="best_pid.txt"):
    """
    Save PID gain values to a file for later loading by the server.

    Format: comma-separated values "Kp,Ki,Kd".
    """
    with open(path, "w") as f:
        f.write(f"{Kp},{Ki},{Kd}")  # Overwrite or create the file


def load_pid(path="best_pid.txt"):
    """
    Load PID gain values from a file, or return default values if the file does not exist.

    Returns:
        Tuple of floats: (Kp, Ki, Kd)
    """
    # If no PID file exists, return default fallback gains
    if not os.path.exists(path):
        return 0.1, 0.01, 0.05
    # Read comma-separated gains from the file
    with open(path) as f:
        # Map the three values to floats
        Kp, Ki, Kd = map(float, f.read().split(","))
    return Kp, Ki, Kd
