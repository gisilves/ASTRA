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
    y_values = y_values[:-int(len(y_values) * 0.1)]*1000

    # Compute the baseline for values before time 0us
    t0_idx = np.abs(x_values - 0).argmin()
    baseline = np.median(y_values[:t0_idx])

    # Find the minimum value of the waveform
    min_y = np.min(y_values)

    # Subtract the baseline from the waveform
    signal = baseline - min_y

    return signal

def plot_waveform_by_cfg(file_path, pt, ax):
        folders = sorted(glob.glob(file_path + '/*'))
        good_folders = []
    
        # Replace '.' for 'e' in the pt string for non integer values
        if pt.is_integer():
            pt_str = '_pt'+str(int(pt))
        else:
            pt_str = '_pt'+str(pt).replace('.', 'e')
        
        for folder in folders:
            if pt_str in folder:
                good_folders.append(folder)
    
        print(good_folders)

        # Vtp values (in fC)
        vtp_values = np.linspace(3.6,280,100)

        # Set the color palette
        colors = sns.color_palette("colorblind", len(good_folders))
        handles = []
        labels = []

        for folder in good_folders:
            # Find and sort all files .trc in the given path
            files = sorted(glob.glob(folder + '/*.trc'))
        
            my_title(ax, 'Peaking time = ' + str(pt) + 'us')

            vtp_idx = 0
            y_values = []
            for file in files:
                print('Processing file: ' + file, flush=True, end='\r')
                signal = process_waveform(file)
                y_values.append(signal)
                ax.plot(vtp_values[vtp_idx], signal, 'o', color=colors[good_folders.index(folder)], label=folder.split('/')[-1])
                vtp_idx += 1

            ax.set_xlabel('Vtp [fC]')
            ax.set_ylabel('Amplitude [mV]')

            # Perform a linear fit: 20 to 100fC if gain is 1, 20 to 150fC if gain is 0
            idx_25fC = np.abs(vtp_values - 25).argmin()
            idx_100fC = np.abs(vtp_values - 100).argmin()
            idx_160fC = np.abs(vtp_values - 160).argmin()

            if 'gain1' in folder:
                m, b = np.polyfit(vtp_values[idx_25fC:idx_100fC], y_values[idx_25fC:idx_100fC], 1)
                ax.plot(vtp_values[idx_25fC:idx_100fC], m*vtp_values[idx_25fC:idx_100fC] + b, color=colors[good_folders.index(folder)])
            else:
                m, b = np.polyfit(vtp_values[idx_25fC:idx_160fC], y_values[idx_25fC:idx_160fC], 1)
                ax.plot(vtp_values[idx_25fC:idx_160fC], m*vtp_values[idx_25fC:idx_160fC] + b, color=colors[good_folders.index(folder)])
            print('Fit: y = ' + str(round(m, 2)) + 'x + ' + str(round(b, 2)))
            
            # Handles and Labels for the legend
            handles.append(mpatches.Patch(color=colors[good_folders.index(folder)]))
            labels.append(folder.split('/')[-1] + '\n y = ' + str(round(m, 2)) + 'x + ' + str(round(b, 2)))
            print()

        # Add the legend
        ax.legend(handles, labels, loc='lower right')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot Vtp scan')

    parser.add_argument('--file_path', type=str, help='Path to the folder containing the folders containing .trc files', default='/media/DATA/', dest='file_path')

    # If no arguments are given, print the help message
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Check if the file path is valid
    if not glob.glob(args.file_path):
        print("Invalid file path")
        sys.exit(1)

    # Loop on folders and find all the different peaking times
    folders = sorted(glob.glob(args.file_path + '/*'))
    pts = []
    for folder in folders:
        if '_pt' in folder:
            pt = folder.split('_pt')[-1]
            if pt not in pts:
                # Replace 'e' for '.' in the pt string and append it as a float
                pts.append(float(pt.replace('e', '.')))
    
    # Remove repeated values
    pts = list(set(pts))

    print('Found the following peaking times:')
    print(pts)


    # Subplots for the different peaking times
    fig, axs = plt.subplots(len(pts), 1, figsize=(12, len(pts)*6.25))
    for i, pt in enumerate(pts):
        plot_waveform_by_cfg(args.file_path, pt, axs[i])    
                          
    plt.savefig('waveforms_by_pt_gain_comparison.png', dpi=300, bbox_inches='tight', pad_inches=0.1)