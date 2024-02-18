import sys
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from scipy.interpolate import UnivariateSpline
from lecroyutils.data import LecroyScopeData

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

if __name__ == '__main__':
    # Check if a file path, start and stop waveforms are provided as command-line arguments
    if len(sys.argv) != 5:
        print("Usage: python plot_vtp_scan.py <file_path> <start_waveform> <stop_waveform> <peaking_time> <fit_min> <fit_max>")
        sys.exit(1)

    # Extract the file path, start and stop waveforms from the command-line arguments 
    file_path = sys.argv[1]
    start_waveform = int(sys.argv[2])
    stop_waveform = int(sys.argv[3])
    peak_time = float(sys.argv[4])


    fit_min = 20
    fit_max = 140

    # Check if the file path is valid
    if not glob.glob(file_path):
        print("Invalid file path")
        sys.exit(1)

    # Find and sort all files .trc in the given path
    files = sorted(glob.glob(file_path + '/*.trc'))
    print('Found ' + str(len(files)) + ' files')

    # Vtp values (inf fC)
    vtp_values = np.linspace(3.6,280,100)

    fig = plt.subplots(2, 1, figsize=(10, 5))
    ax = plt.subplot(2, 1, 1)
    # Shared x axis
    plt.subplots_adjust(hspace=0.5)
    

    # Loop from start_waveform to stop_waveform
    peak_values = []

    vtp_idx = 0
    for waveform in range(start_waveform, stop_waveform):
        # Process the waveform
        print('Processing waveform ' + str(waveform), flush=True, end='\r')
        if waveform >= 299:
            waveform -= 100
        x_values, y_values = process_waveform(files[waveform])

        # Find the value of the peak at peak_time
        idx_peak = np.abs(x_values - peak_time).argmin()
        peak_values.append((vtp_values[vtp_idx], y_values[idx_peak]))

        # Plot the data (add the corresponding Vtp value as label)
        ax.plot(x_values, y_values, label='Vtp = ' + str(vtp_values[vtp_idx]) + 'fC')
        vtp_idx += 1


    # Add a title
    ax.set_title('Vtp scan ' + str(start_waveform) + ' to ' + str(stop_waveform))
    # Add x and y labels
    ax.set_xlabel('Time (us)')
    # Add x divisions every 10 us
    ax.set_xticks(np.arange(0, 100, 10))
    ax.set_ylabel('Amplitude (V)')

    # Add a line at peak_time
    ax.axvline(x=peak_time, color='r', linestyle='--')

    # Add light grid
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # Second plot for the peak values
    ax2 = plt.subplot(2, 1, 2)

    # Plot the peak values at 10 us
    peak_values_at_10 = np.array(peak_values)
    ax2.plot(peak_values_at_10[:, 0], (1.2 - peak_values_at_10[:, 1])*100, 'o') # NOTE: 1.2V should be the baseline as set by BLH voltage
    ax2.set_title('Peak values at ' + str(peak_time) + ' us')
    ax2.set_xlabel('Vtp (fC)')
    # Add x divisions every 10 fC
    ax2.set_xticks(np.arange(0, 300, 10)) 
    ax2.set_ylabel('Peak value (mV)')

    # Add light grid
    ax2.grid(color='gray', linestyle='--', linewidth=0.5)

    # Linear regression from fit_min to fit_max fC
    idx_20 = np.abs(peak_values_at_10[:, 0] - fit_min).argmin()
    idx_140 = np.abs(peak_values_at_10[:, 0] - fit_max).argmin()

    x = peak_values_at_10[idx_20:idx_140, 0]
    y = (1.2 - peak_values_at_10[idx_20:idx_140, 1])*100
    m, b = np.polyfit(x, y, 1)
    ax2.plot(x, m*x + b, label='y = ' + str(m) + 'x + ' + str(b))
    ax2.legend()
 
 
    plt.show()

    
    