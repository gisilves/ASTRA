import sys
import argparse
import functions  # Ensure functions.py is in the same directory or in the PYTHONPATH
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Use seaborn for plotting
sns.set(style="whitegrid")
# Use bigger fonts
sns.set_context("paper", font_scale=1.5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Vtp scan')
    parser.add_argument('--file_path', type=str, help='Path to the folder containing the .trc files', default='/media/DATA/', dest='file_path')
    parser.add_argument('--vblh', type=int, help='VBLH value', required=True, dest='vblh')
    parser.add_argument('--start_waveform', type=int, help='Start waveform', default=0, dest='start_waveform')
    parser.add_argument('--stop_waveform', type=int, help='Stop waveform', default=99, dest='stop_waveform')
    parser.add_argument('--peaking_time', type=float, help='Nominal peaking time of the setup', default=9, dest='peak_time')
    parser.add_argument('--fit_start_peak', type=float, help='Minimum value for the linear fit', default=20, dest='fit_start_peak')
    parser.add_argument('--fit_end_peak', type=float, help='Maximum value for the linear fit', default=140, dest='fit_end_peak')
    parser.add_argument('--fit_start_min', type=float, help='Minimum value for the linear fit', default=20, dest='fit_start_min')
    parser.add_argument('--fit_end_min', type=float, help='Maximum value for the linear fit', default=140, dest='fit_end_min')
    parser.add_argument('--auto_fit', action='store_true', help='Automatically find the fit range', default=False, dest='auto_fit')
    parser.add_argument('--positive_waveforms', action='store_true', help='Assume positive waveforms', default=False, dest='positive_waveforms')

    # If no arguments are given, print the help message
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    folders = sorted(os.listdir(args.file_path))
    gains_PT = []
    gains_true = []
    max_linear = []
    ideal_peaking_time = []
    baselines = []

    print('Processing Vtp scan with following parameters:')
    print(f'\tFile path: {args.file_path}')
    print(f'\tVBLH: {args.vblh}')
    print(f'\tStart waveform: {args.start_waveform}')
    print(f'\tStop waveform: {args.stop_waveform}')
    print(f'\tPeaking time: {args.peak_time}')
    print(f'\tFit min at peak: {args.fit_start_peak}')
    print(f'\tFit max at peak: {args.fit_end_peak}')
    print(f'\tFit min at minimum: {args.fit_start_min}')
    print(f'\tFit max at minimum: {args.fit_end_min}')
    print(f'\tAuto fit: {args.auto_fit}')
    print(f'\tPositive waveforms: {args.positive_waveforms}')

    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Loop on all the folders
    for folder in folders:
        folder_path = os.path.join(args.file_path, folder, f'VBLH{args.vblh}')
        channel = int(folder.split('_CH')[1])
        print(f'Processing folder {folder_path}/')

        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f'Folder {folder_path} does not exist')
            continue

        results = functions.process_folder(
            folder_path,
            args.start_waveform, args.stop_waveform,
            args.peak_time, args.fit_start_peak, args.fit_end_peak,
            args.fit_start_min, args.fit_end_min, args.auto_fit,
            False, '', args.positive_waveforms, False
        )
        gains_PT.append(results[0])
        gains_true.append(results[4])
        max_linear.append(results[8])
        ideal_peaking_time.append(results[9])
        baselines.append(np.mean(results[10]))

    titles = ['Gain at peaking time', 'Gain at true peak', 'Computed linear range', 'Ideal peaking time', 'Average Baseline']
    data = [gains_PT, gains_true, max_linear, ideal_peaking_time, baselines]

    y_labels = ['Gain', 'Gain', 'Max linear range', 'Ideal peaking time', 'Average Baseline']
    filenames = ['gain_PT', 'gain_true', 'max_linear', 'ideal_peaking_time', 'baseline']

    i = 0
    for title, y_label, filename in zip(titles, y_labels, filenames):
        plt.figure(figsize=(12, 6))
        plt.plot(data[i], 'o', label=f'VBLH = {args.vblh}')
        plt.xlabel('Channel')
        plt.ylabel(y_label)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        functions.my_title(plt.gca(), title)
        plt.legend()
        plt.savefig(f'plots/{filename}_VBLH{args.vblh}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        i += 1

    # Save the raw data in a txt file
    np.savetxt(f'plots/gain_distro_VBLH{args.vblh}.txt', np.column_stack((gains_PT, gains_true, max_linear, ideal_peaking_time, baselines)),
               fmt='%f', delimiter=',', header='Gain at PT, Gain at true, Max linear range, Ideal peaking time, Average Baseline')

    # Create and save histogram plots
    i = 0
    for title, y_label, filename in zip(titles, y_labels, filenames):
        plt.figure(figsize=(12, 6))
        plt.hist(data[i], bins=10, histtype='step', label=f'VBLH = {args.vblh}', linewidth=2, range=(np.min(data[i]), np.max(data[i])))
        plt.xlabel(y_label)
        plt.ylabel('Counts')
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        functions.my_title(plt.gca(), title)
        textstr = '\n'.join((f'$\\mu={np.mean(data[i]):.2f}$', f'$\\sigma={np.std(data[i]):.2f}$'))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', bbox=props)
        plt.legend(loc='upper right')
        plt.savefig(f'plots/{filename}_hist_VBLH{args.vblh}.png', format='png', dpi=300, bbox_inches='tight')
        plt.close()
        i += 1

