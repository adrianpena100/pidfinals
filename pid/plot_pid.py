import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv

rounds, losses, P_vals, I_vals, D_vals, pid_vals = [], [], [], [], [], []
attack_rounds = set()

# Load attack round log
try:
    with open("malicious_rounds.txt", "r") as f:
        next(f)  # skip header
        for line in f:
            attack_rounds.add(int(line.strip()))
except FileNotFoundError:
    print("malicious_rounds.txt not found")

# Load PID controller logs
with open("pid_log.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rnd = int(row["round"])
        rounds.append(rnd)
        losses.append(float(row["avg_loss"]))
        P_vals.append(float(row["P"]))
        I_vals.append(float(row["I"]))
        D_vals.append(float(row["D"]))
        pid_vals.append(float(row["pid_output"]))

# Plot results
plt.figure(figsize=(12, 6))
for rnd in rounds:
    if rnd in attack_rounds:
        plt.axvspan(rnd - 0.5, rnd + 0.5, color="red", alpha=0.15)

plt.plot(rounds, losses, label="Average Loss", linewidth=2)
plt.plot(rounds, pid_vals, label="PID Output", linestyle="--")
plt.plot(rounds, P_vals, label="P Term", linestyle=":")
plt.plot(rounds, I_vals, label="I Term", linestyle=":")
plt.plot(rounds, D_vals, label="D Term", linestyle=":")

plt.xlabel("Round")
plt.ylabel("Value")
plt.title("PID Dynamics with Malicious Rounds Highlighted")
plt.legend()
plt.grid(True)

# Show every round number on the x-axis
plt.xticks(range(min(rounds), max(rounds) + 1))

plt.tight_layout()
plt.savefig("pid_plot_with_attacks.png")
plt.show()
