import sys
import glob
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline
from lecroyutils.data import LecroyScopeData

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

def auto_fit_range(data, min):
    # Find automatically the maximum of the fit range (assuming the ASIC is linear in the first part of the scan)

    # Fit from fit_start_peak to 40fc
    idx_min = np.abs(data[:, 0] - min).argmin()
    idx_max = np.abs(data[:, 0] - 40).argmin()
    x = data[idx_min:idx_max, 0]
    y = (data[idx_min:idx_max,2] - data[idx_min:idx_max, 1])
    linearfit_start = linregress(x, y)
    m_s = linearfit_start.slope
    b_S = linearfit_start.intercept

    # Fit from 200fc to 240fc
    idx_min = np.abs(data[:, 0] - 200).argmin()
    idx_max = np.abs(data[:, 0] - 240).argmin()
    x = data[idx_min:idx_max, 0]
    y = (data[idx_min:idx_max,2] - data[idx_min:idx_max, 1])
    linearfit_end = linregress(x, y)
    m_e = linearfit_end.slope
    b_e = linearfit_end.intercept

    # Find the intersection of the two lines
    fit_max = (b_e - b_S) / (m_s - m_e)    
    print('Automatically computed fit max for peak values: ' + str(fit_max))
    return fit_max

def process_folder(file_path, start_waveform, stop_waveform, peaking_time, fit_start_peak, fit_end_peak, fit_start_min, fit_end_min, auto_fit, pt_line, note):
    # Check if the file path is valid
    if not glob.glob(file_path):
        print("Invalid file path")
        sys.exit(1)

    # Find and sort all files .trc in the given path
    files = sorted(glob.glob(file_path + '/*.trc'))
    print('Found ' + str(len(files)) + ' files')

    # Vtp values (inf fC)
    vtp_values = np.linspace(3.6,280,100)

    # Loop from start_waveform to stop_waveform
    peak_values = []
    min_values = []

    fig = plt.subplots(4, 1, figsize=(12, 25))
    ax = plt.subplot(4, 1, 1)

    vtp_idx = 0
    for waveform in range(start_waveform, stop_waveform + 1):
        # Process the waveform
        print('Processing waveform ' + str(waveform), flush=True, end='\r')
        if waveform >= 299:
            waveform -= 100
        x_values, y_values = process_waveform(files[waveform])

        # Find the time at time 0 us
        idx_0 = np.abs(x_values).argmin()
        
        # Compute the baseline up to 0 us
        baseline = np.median(y_values[:idx_0])*1000

        # Find the value of the peak at peak_time
        idx_peak = np.abs(x_values - peaking_time).argmin()
        peak_values.append((vtp_values[vtp_idx], y_values[idx_peak]*1000, baseline))

        # Find the time of the minimum value
        idx_min = np.abs(y_values).argmin()
        min_values.append((vtp_values[vtp_idx], x_values[idx_min], y_values[idx_min]*1000, baseline))

        # Plot the data (add the corresponding Vtp value as label) using seaborn
        ax.plot(x_values, y_values, label='Vtp = ' + str(vtp_values[vtp_idx]) + 'fC')
        vtp_idx += 1

    # Add a title
    my_title(ax,'Vtp scan ' + str(round(vtp_values[start_waveform],1)) + 'fC to ' + str(round(vtp_values[stop_waveform],1)) + 'fC ')
    
    # Add labels
    ax.set_xlabel('Time (us)', loc='right')
    ax.set_ylabel('Amplitude (V)', loc='top')
    # Add x divisions every 10 us
    ax.set_xticks(np.arange(0, 100, 10))

    if pt_line:
        # Add a line at peak_time
        ax.axvline(x=peaking_time, color='r', linestyle='--')

    # Add light grid
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    # Second plot for the peak values
    ax2 = plt.subplot(4, 1, 2)

    # Plot the peak values at peak_time
    peak_value_at_PT = np.array(peak_values)
    ax2.plot(peak_value_at_PT[:, 0], (peak_value_at_PT[:,2] - peak_value_at_PT[:, 1]), 'o') # NOTE: 1.2V should be the baseline as set by BLH voltage
    my_title(ax2, 'Peak values at ' + str(peaking_time) + ' us')
    ax2.set_xlabel('Vtp (fC)', loc='right')
    # Add x divisions every 10 fC
    ax2.set_xticks(np.arange(0, 300, 10)) 
    # Rotate slightly the x labels
    for tick in ax2.get_xticklabels():
        tick.set_rotation(45)
    ax2.set_ylabel('Peak amplitude value (mV)', loc='top')

    # Add light grid
    ax2.grid(color='gray', linestyle='--', linewidth=0.5)

    if auto_fit:
        fit_end_peak = auto_fit_range(peak_value_at_PT, fit_start_peak)

    # Linear regression from fit_min to fit_max fC
    idx_min = np.abs(peak_value_at_PT[:, 0] - fit_start_peak).argmin()
    idx_max = np.abs(peak_value_at_PT[:, 0] - fit_end_peak).argmin()

    x = peak_value_at_PT[idx_min:idx_max, 0]
    y = (peak_value_at_PT[idx_min:idx_max,2] - peak_value_at_PT[idx_min:idx_max, 1])
    linearfit = linregress(x, y)
    m = linearfit.slope
    b = linearfit.intercept
    ax2.plot(x, m*x + b, label='y = ' + str(m) + 'x + ' + str(b))
    # Print linear fit equation (up to two decimal places)
    ax2.text(0.75, 0.25, 'y = ' + str(round(m, 2)) + 'x + ' + str(round(b, 2)), 
        horizontalalignment='center', 
        verticalalignment='center', 
        transform=ax2.transAxes,         
        family='monospace',
        style='italic',
        weight='bold')

    # Plot the minimum time values wrt vtp
    ax3 = plt.subplot(4, 1, 3)
    min_values = np.array(min_values)
    ax3.plot(vtp_values[start_waveform:stop_waveform+1], min_values[:, 1], 'o')

    # Add labels
    ax3.set_xlabel('Vtp (mV)', loc='right')
    ax3.set_ylabel('Time @ minimum value (us)', loc='top')
    # Add x divisions every 10 us
    ax3.set_xticks(np.arange(0, 300, 10))
    # Rotate slightly the x labels
    for tick in ax3.get_xticklabels():
        tick.set_rotation(45)
    my_title(ax3, 'True peaking time')

    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(vtp_values[start_waveform:stop_waveform+1], min_values[:, 3] - min_values[:, 2], 'o')

    if auto_fit:
        fit_end_min = auto_fit_range(np.delete(min_values,1,1), fit_start_min)

    # Linear regression from fit_min to fit_max fC
    idx_min = np.abs(vtp_values - fit_start_min).argmin()
    idx_max = np.abs(vtp_values - fit_end_min).argmin()
    x = vtp_values[idx_min:idx_max]
    y = min_values[idx_min:idx_max, 3] - min_values[idx_min:idx_max, 2]

    # Perform linear fit with scipy
    linearfit = linregress(x, y)
    m = linearfit.slope
    b = linearfit.intercept
    pvalue = linearfit.pvalue
    rvalue = linearfit.rvalue
    
    ax4.plot(x, m*x + b, label='y = ' + str(m) + 'x + ' + str(b))

    # Print linear fit equation (up to two decimal places)
    ax4.text(0.75, 0.25, 'y = ' + str(round(m, 2)) + 'x + ' + str(round(b, 2)), 
        horizontalalignment='center', 
        verticalalignment='center', 
        transform=ax4.transAxes,         
        family='monospace',
        style='italic',
        weight='bold')
    
    ax4.set_xlabel('Vtp (fC)', loc='right')
    # Add x divisions every 10 fC
    ax4.set_xticks(np.arange(0, 300, 10))
    # Rotate slightly the x labels
    for tick in ax4.get_xticklabels():
        tick.set_rotation(45)
    ax4.set_ylabel('Minimum value amplitude (mV)', loc='top')
    my_title(ax4, 'Minimum value amplitude vs Vtp')

    if pt_line:
        plt.savefig('plots/vtp_scan_amp_'+file_path.split('/')[-1]+note+'.png', format='png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('plots/vtp_scan_no_pt_line_amp_'+file_path.split('/')[-1]+note+'.png', format='png', dpi=300, bbox_inches='tight')