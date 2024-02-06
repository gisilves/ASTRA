import sys
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from lecroyutils.data import LecroyScopeData

def process_waveform(file_path, ax_original, points, peaking_time_us=10):
    # Load data from the file
    data = LecroyScopeData.parse_file(file_path)

    # Get x and y values from the loaded data
    x_values = data.x
    y_values = data.y

    # Convert x_values to seconds (total width of the waveform is 100 us)
    x_values = x_values * 1e6

    # Remove the last 10% of the data to avoid exclude the positive peak
    x_values = x_values[:-int(len(x_values) * 0.1)]
    y_values = y_values[:-int(len(y_values) * 0.1)]

    # Retrieve tes pulse values from name
    test_pulse = int(file_path.split('/')[-2].split('_')[0].split('p')[1])

    # Plot the data (add filename as label)
    ax_original[0].plot(x_values, y_values, label='Vtp = ' + str(test_pulse) + 'mV')

    # Find the x idx closest to 10
    idx_peak = np.abs(x_values - peaking_time_us).argmin()
    points.append((test_pulse*0.24, 1 - y_values[idx_peak]))

    # Add a single point to ax_original[1]
    #ax_original[1].plot(, y_values[idx_10], 'o')

if __name__ == '__main__':
    # Check if a file path and thread count are provided as command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python plot_waveform_by_config.py <file_path> <cfg_num> <peaking_time_us>")
        sys.exit(1)

    # Extract the file path and config number from the command-line arguments
    file_path = sys.argv[1]
    cfg_num = int(sys.argv[2])
    peaking_time_us = int(sys.argv[3])

    # Check if the config number is valid
    if not (0 <= cfg_num <= 1792):
        print("Config not available")
        sys.exit(1)

    # Check if the file path is valid
    if not glob.glob(file_path):
        print("Invalid file path")
        sys.exit(1)

    # Find all the folders in the given path
    folders = sorted(glob.glob(file_path + '/*'))

    # Figure and subplots for plotting
    fig, ax_original = plt.subplots(2, 1, figsize=(16, 9))

    points = []

    # Process the waveform files in plotting them on separate subplots
    for folder in folders:
        file = f"{folder}/C1--{folder.split('/')[-1]}--{str(cfg_num).zfill(5)}.trc"
        print(file)
        process_waveform(file, ax_original, points, peaking_time_us)

    ax_original[0].set_ylabel('Y values (V)')
    ax_original[0].set_xlabel('X values (us)')
    ax_original[0].set_title('Config ' + str(cfg_num))
    ax_original[0].legend()

    points = np.array(points)
    ax_original[1].plot(points[:, 0], points[:, 1], 'o')
    # Linear regression
    m, b = np.polyfit(points[:, 0], points[:, 1], 1)
    ax_original[1].plot(points[:, 0], m*points[:, 0] + b, color='red')

    # Extend the line for x from 0 to the first point and from the last point to 250
    ax_original[1].plot([0, points[0, 0]], [b, m*points[0, 0] + b], '--', color='red')
    ax_original[1].plot([points[-1, 0], 250], [m*points[-1, 0] + b, m*250 + b], '--', color='red')

    # Add line equation to the plot on the top left corner of the second plot (m as scientific notation)
    ax_original[1].text(0.1, 0.95, f"y = {m:.3e}x + {b:.2f}", transform=ax_original[1].transAxes)

    ax_original[1].set_ylabel('Value at ' + str(peaking_time_us) + 'us (inverted)')
    ax_original[1].set_xlabel('Test Pulse (fC)')

    plt.tight_layout()
    folder_name = file_path.split('/')[-1]

    plt.savefig(f"waveform_by_config_{folder_name}_{cfg_num}_{peaking_time_us}us.png", dpi=300)
