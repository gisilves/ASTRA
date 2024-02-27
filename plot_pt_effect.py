import sys
import glob
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from lecroyutils.data import LecroyScopeData


# Use seaborn for plotting
sns.set(style="whitegrid")
# Use bigger fonts
sns.set_context("paper", font_scale=1.5)

def my_title(ax, title):
    ax.text(1.02, 0.5, title,
        horizontalalignment='center',
        verticalalignment='center_baseline',
        rotation=-90,
        rotation_mode='anchor',
        transform_rotates_text=True,
        transform=ax.transAxes,
        family='monospace',
        style='italic',
        weight='bold')

def process_waveform(file_path):
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

    return np.array([x_values, y_values])

def plot_waveform_by_cfg(file_path, vtp, gain, ax, min_y=0.3):
        folders = sorted(glob.glob(file_path + '/*'))
        good_folders = []

        # Find all folders with names in the format Vtp_gainX_ptY
        for folder in folders:
            if 'Vtp_gain'+str(gain) in folder:
                good_folders.append(folder)

        # Vtp values (inf fC)
        vtp_values = np.linspace(3.6,280,100)

        # Find the index of the Vtp value in input
        vtp_idx = np.abs(vtp_values - vtp).argmin()

        # Loop over the good folders and find the corresponding .trc file
        good_files = []

        for folder in good_folders:
            # Find and sort all files .trc in the given path
            files = sorted(glob.glob(folder + '/*.trc'))
            if len(files) == 0:
                # Remove folder from the list
                good_folders.remove(folder)

        # Loop on remaining folders and find the corresponding .trc file
        for folder in good_folders:
            # Find and sort all files .trc in the given path
            files = sorted(glob.glob(folder + '/*.trc'))
            good_files.append(files[vtp_idx])

        
        my_title(ax, 'Vtp = ' + str(vtp) + ' fC - Gain = ' + str(gain))
        for file in good_files:
            x_values, y_values = process_waveform(file)
            ax.plot(x_values, y_values, label=file.split('/')[-2])
        ax.set_xlabel('Time [us]')
        ax.set_ylabel('Voltage [V]')

        # Set scale limits
        ax.set_xlim(-10, 80)
        ax.set_ylim(min_y, 1.2)

        handles, labels = ax.get_legend_handles_labels()
        # Remove string before pt in all labels
        labels = [label.split('_')[-1] for label in labels]
        # Swap 'e' for '.' in all labels
        labels = [label.replace('e', '.') for label in labels]
        ax.legend(handles, labels, loc='lower right')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot Vtp scan')

    parser.add_argument('--file_path', type=str, help='Path to the folder containing the folders containing .trc files', default='/media/DATA/', dest='file_path')
    parser.add_argument('--vtp', type=int, help='Value for the Vtp', dest='vtp_value', required=True)
    parser.add_argument('--min_y', type=float, help='Minimum value for the y axis', dest='min_y', default=0.3)
    
    # If no arguments are given, print the help message
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    print('Processing Vtp scan with following parameters:')
    print('File path: ' + args.file_path)
    print('Vtp value: ' + str(args.vtp_value))

    # Check if the file path is valid
    if not glob.glob(args.file_path):
        print("Invalid file path")
        sys.exit(1)

    # Process the waveforms
    fig = plt.subplots(1, 2, figsize=(16, 9))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    plot_waveform_by_cfg(args.file_path, args.vtp_value, 0, ax1, args.min_y)
    plot_waveform_by_cfg(args.file_path, args.vtp_value, 1, ax2, args.min_y)

    plt.savefig('Vtp_' + str(args.vtp_value) + '_by_gain.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()