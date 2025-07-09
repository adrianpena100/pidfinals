import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv

rounds, P_vals, I_vals, D_vals, pid_vals = [], [], [], [], []

# Load PID controller logs
with open("pid_log.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rnd = int(row["round"])
        rounds.append(rnd)
        P_vals.append(float(row["P"]))
        I_vals.append(float(row["I"]))
        D_vals.append(float(row["D"]))
        pid_vals.append(float(row["pid_output"]))

# Plot P, I, D, and PID output
plt.figure(figsize=(10, 6))
plt.plot(rounds, P_vals, label="P Term", linestyle=":", linewidth=2)
plt.plot(rounds, I_vals, label="I Term", linestyle="-.", linewidth=2)
plt.plot(rounds, D_vals, label="D Term", linestyle="--", linewidth=2)
plt.plot(rounds, pid_vals, label="PID Output", linestyle="-", linewidth=2, color="black")

plt.xlabel("Round", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.title("PID Controller Terms per Round", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle=":", linewidth=0.7)

plt.tight_layout()
plt.savefig("pid_terms_plot.png")
plt.show()
