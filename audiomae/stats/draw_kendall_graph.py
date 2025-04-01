import matplotlib.pyplot as plt
import torch
import numpy as np
import json
import argparse

# get the first cmdline argument
parser = argparse.ArgumentParser()
parser.add_argument("filename", help="the filename of the json file")
# get the second argument as a title
parser.add_argument("title", help="the title of the graph")
args = parser.parse_args()

filename = args.filename

data = json.load(open(f'{filename}', 'r'))
print(data)


# use gpt

# X-axis values (1 to 12)

x_values = np.arange(1, 13)


# Plotting

plt.figure(figsize=(10, 2))

for label, y_values in data.items():

    avg_val = np.mean(y_values)
    
    line_obj = plt.plot(x_values, y_values, label=f"{label} / {avg_val:.2f}", marker='o')
    
    # Scatter plot at x=13 (or wherever you'd like)
    # Use the same color as the line:
    color = line_obj[0].get_color()
    
    print(label)
    # plt.scatter(5.5, avg_val, color=color, marker='D', s=60)
    
    

# Y-axis limits and labels

plt.ylim(0.0, 1.0)

# plt.xlabel(f"{args.title} - Block ID", fontsize=16)

plt.ylabel("AST" "\n" r"$\tau$ - mean & attn", fontsize=16)


# plt.title("Line Graphs of Different Datasets")

plt.xticks(x_values, fontsize=16)  # Ensure x-axis ticks correspond to the data points
plt.yticks(fontsize=16)

plt.legend(fontsize=16, loc='lower right', ncol=3)

plt.grid(alpha=0.3)

# Show the plot

plt.tight_layout()

plt.savefig(f'{filename}.jpg')
plt.savefig(f'{filename}.eps', format='eps')
