import sys
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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


import matplotlib.gridspec as gridspec

if __name__ == '__main__':
    # Check if a file path and thread count are provided as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python plot_waveform_by_config.py <file_path> <cfg_num>")
        sys.exit(1)

    # Extract the file path and config number from the command-line arguments
    file_path = sys.argv[1]
    cfg_num = int(sys.argv[2])

    # Check if the config number is valid
    if not (0 <= cfg_num <= 1792):
        print("Config not available")
        sys.exit(1)

    # Check if the file path is valid
    if not glob.glob(file_path):
        print("Invalid file path")
        sys.exit(1)

    # Find all the folders for low BLH in the given path (main folder is AVG_L_BLH)
    folders_low = sorted(glob.glob(file_path + '/AVG_L_BLH/*'))
    # Find all the folders for high BLH in the given path (main folder is AVG_H_BLH)
    folders_high = sorted(glob.glob(file_path + '/AVG_H_BLH/*'))
    
    print(folders_low)
    print(folders_high)

    # Each folder contains a scan for several Vtp values (VtpXXXX_BLH_(L or H)_AVG): find the common Vtp values for both low and high BLH
    vtp_values_low = [int(folder.split('Vtp')[1].split('_')[0]) for folder in folders_low]
    vtp_values_high = [int(folder.split('Vtp')[1].split('_')[0]) for folder in folders_high]
    vtp_values = sorted(list(set(vtp_values_low).intersection(vtp_values_high)))

    # Create a PdfPages object to save plots to a PDF file
    pdf_pages = PdfPages('waveform_plots.pdf')

    # Loop over the common Vtp values and plot the waveforms for low and high BLH for the given config number
    for vtp in vtp_values:
        # Figure and subplots for plotting
        fig, ax_original = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
        plt.subplots_adjust(hspace=0)  # Ensure subplots touch
        fig.suptitle('Vtp = ' + str(vtp) + 'mV, config ' + str(cfg_num))

        file_low = f"{file_path}/AVG_L_BLH/Vtp{str(vtp).zfill(4)}_BLH_L_AVG/C1--Vtp{str(vtp).zfill(4)}_BLH_L_AVG--{str(cfg_num).zfill(5)}.trc"
        points_low = process_waveform(file_low)
        file_high = f"{file_path}/AVG_H_BLH/Vtp{str(vtp).zfill(4)}_BLH_H_AVG/C1--Vtp{str(vtp).zfill(4)}_BLH_H_AVG--{str(cfg_num).zfill(5)}.trc"
        points_high = process_waveform(file_high)

        # Plot the data (add filename as label)
        ax_original[0].plot(points_low[0], points_low[1], label='Low BLH')
        ax_original[0].plot(points_high[0], points_high[1], label='High BLH')
        ax_original[0].set_ylabel('Y values (V)')
        ax_original[0].legend()

        difference = points_high[1] - points_low[1]

        ax_original[1].plot(points_low[0], difference, label='High - Low BLH')
        ax_original[1].set_ylabel('Y values (V)')
        ax_original[1].set_xlabel('X values (us)')

        # Save the current figure to the PDF with tight layout
        pdf_pages.savefig(fig, bbox_inches='tight')

    # Close the PdfPages object
    pdf_pages.close()