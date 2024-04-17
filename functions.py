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
import time

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
    return fit_max - 10

def loop_on_waveforms(vtp_values, files, start_waveform, stop_waveform, peaking_time, ax, pt_line, plotting):

    # Loop from start_waveform to stop_waveform
    peak_values = []
    min_values = []
    max_values = []

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
        idx_max = np.abs(y_values).argmax()
        min_values.append((vtp_values[vtp_idx], x_values[idx_min], y_values[idx_min]*1000, baseline))
        max_values.append((vtp_values[vtp_idx], x_values[idx_max], y_values[idx_max]*1000, baseline))

        if plotting:
            # Plot the data (add the corresponding Vtp value as label) using seaborn
            ax.plot(x_values, y_values, label='Vtp = ' + str(vtp_values[vtp_idx]) + 'fC')
        vtp_idx += 1

    if plotting:    
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

    # Return the peak values, min values and max values as numpy arrays
    return np.array(peak_values), np.array(min_values), np.array(max_values)

def values_at_peaking_time(ax, peak_values, peaking_time, fit_start_peak, fit_end_peak, auto_fit, plotting):
    # Plot the peak values at peak_time
    peak_value_at_PT = np.array(peak_values)

    if plotting:
        ax.plot(peak_value_at_PT[:, 0], np.abs(peak_value_at_PT[:,2] - peak_value_at_PT[:, 1]), 'o') # NOTE: 1.2V should be the baseline as set by BLH voltage
        my_title(ax, 'Peak values at ' + str(peaking_time) + ' us')
        ax.set_xlabel('Vtp (fC)', loc='right')
        # Add x divisions every 10 fC
        ax.set_xticks(np.arange(0, 300, 10)) 
        # Rotate slightly the x labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.set_ylabel('Peak amplitude value (mV)', loc='top')

        # Add light grid
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

    if auto_fit:
        fit_end_peak = auto_fit_range(peak_value_at_PT, fit_start_peak)

    # Linear regression from fit_start_peak to fit_end_peak fC
    idx_min = np.abs(peak_value_at_PT[:, 0] - fit_start_peak).argmin()
    idx_max = np.abs(peak_value_at_PT[:, 0] - fit_end_peak).argmin()

    x = peak_value_at_PT[idx_min:idx_max, 0]
    y = np.abs(peak_value_at_PT[idx_min:idx_max,2] - peak_value_at_PT[idx_min:idx_max, 1])
    linearfit = linregress(x, y)
    m_peak = linearfit.slope
    b_peak = linearfit.intercept
    pvalue_peak = linearfit.pvalue
    rvalue_peak = linearfit.rvalue

    if plotting:
        ax.plot(x, m_peak*x + b_peak, label='y = ' + str(m_peak) + 'x + ' + str(b_peak))
        # Print linear fit equation (up to two decimal places)
        ax.text(0.75, 0.25, 'y = ' + str(round(m_peak, 2)) + 'x + ' + str(round(b_peak, 2)), 
            horizontalalignment='center', 
            verticalalignment='center', 
            transform=ax.transAxes,         
            family='monospace',
            style='italic',
            weight='bold')
    
    return m_peak, b_peak, pvalue_peak, rvalue_peak, fit_end_peak

def time_at_true_peak(ax, vtp_values, min_values, max_values, start_waveform, stop_waveform, positive_waveforms, max_linear, plotting):
    min_values = np.array(min_values)
    max_values = np.array(max_values)

    # Compute the average time at the true peak from 0 to max_linear fC
    if positive_waveforms:
        idx_max = np.abs(max_values[:, 0] - max_linear).argmin()
        ideal_peaking_time = np.mean(max_values[0:idx_max, 1])
    else:
        idx_max = np.abs(min_values[:, 0] - max_linear).argmin()
        ideal_peaking_time = np.mean(min_values[0:idx_max, 1])

    print('Computed ideal PT: ' + str(ideal_peaking_time))

    if plotting:
        if positive_waveforms:
            ax.plot(vtp_values[start_waveform:stop_waveform+1], max_values[:, 1], 'o')
        else:
            ax.plot(vtp_values[start_waveform:stop_waveform+1], min_values[:, 1], 'o')

        # Add a line at the ideal peaking time
        ax.plot([0, max_linear], [ideal_peaking_time, ideal_peaking_time], 'r--')

        # Add light grid
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

        # Add labels
        ax.set_xlabel('Vtp (mV)', loc='right')
        if positive_waveforms:
            ax.set_ylabel('Time @ true maximum value (us)', loc='top')
        else:
            ax.set_ylabel('Time @ true minimum value (us)', loc='top')
        # Add x divisions every 10 us
        ax.set_xticks(np.arange(0, 300, 10))
        # Rotate slightly the x labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        my_title(ax, 'True peaking time')
        
    return ideal_peaking_time

def amplitude_at_true_peak(ax, vtp_values, min_values, max_values, start_waveform, stop_waveform, positive_waveforms, auto_fit, fit_start_true, fit_end_true, plotting):
    
    if plotting:
        if positive_waveforms:
            ax.plot(vtp_values[start_waveform:stop_waveform+1], np.abs(max_values[:, 3] - max_values[:, 2]), 'o')
        else:
            ax.plot(vtp_values[start_waveform:stop_waveform+1], np.abs(min_values[:, 3] - min_values[:, 2]), 'o')

    if auto_fit:
        if positive_waveforms:
            fit_end_true = auto_fit_range(np.delete(max_values,1,1), fit_start_true)
        else:
            fit_end_true = auto_fit_range(np.delete(min_values,1,1), fit_start_true)

    # Linear regression from fit_start_true to fit_max fC
    idx_min = np.abs(vtp_values - fit_start_true).argmin()
    idx_max = np.abs(vtp_values - fit_end_true).argmin()
    x = vtp_values[idx_min:idx_max]
    if positive_waveforms:
        y = np.abs(max_values[idx_min:idx_max, 3] - max_values[idx_min:idx_max, 2])
    else:
        y = np.abs(min_values[idx_min:idx_max, 3] - min_values[idx_min:idx_max, 2])

    linearfit = linregress(x, y)
    m_true = linearfit.slope
    b_true = linearfit.intercept
    pvalue_true = linearfit.pvalue
    rvalue_true = linearfit.rvalue

    if plotting:
        ax.plot(x, m_true*x + b_true, label='y_true = ' + str(m_true) + 'x + ' + str(b_true))

        # Add light grid
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

        # Print linear fit equation (up to two decimal places)
        ax.text(0.75, 0.25, 'y = ' + str(round(m_true, 2)) + 'x + ' + str(round(b_true, 2)), 
            horizontalalignment='center', 
            verticalalignment='center', 
            transform=ax.transAxes,         
            family='monospace',
            style='italic',
            weight='bold')
        
        ax.set_xlabel('Vtp (fC)', loc='right')
        # Add x divisions every 10 fC
        ax.set_xticks(np.arange(0, 300, 10))
        # Rotate slightly the x labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        if positive_waveforms:
            ax.set_ylabel('Maximum value amplitude (mV)', loc='top')
            my_title(ax, 'Maximum value amplitude vs Vtp')
        else:
            ax.set_ylabel('Minimum value amplitude (mV)', loc='top')
            my_title(ax, 'Minimum value amplitude vs Vtp')

    return m_true, b_true, pvalue_true, rvalue_true, fit_end_true

def process_folder(file_path, start_waveform, stop_waveform, peaking_time, fit_start_peak, fit_end_peak, fit_start_true, fit_end_true, auto_fit, pt_line, note, positive_waveforms, plotting=True):
    # Check if the file path is valid
    if not glob.glob(file_path):
        print("Invalid file path")
        sys.exit(1)

    # Find and sort all files .trc in the given path
    files = sorted(glob.glob(file_path + '/*.trc'))
    print('\tFound ' + str(len(files)) + ' files\n')

    if len(files) == 0:
        print('No files found in the given path')
        return
    
    # Vtp values (inf fC)
    vtp_values = np.linspace(3.6,280,100)

    if plotting:
        fig = plt.subplots(4, 1, figsize=(12, 25))
        ax = plt.subplot(4, 1, 1)
        ax2 = plt.subplot(4, 1, 2)
        ax3 = plt.subplot(4, 1, 3)
        ax4 = plt.subplot(4, 1, 4)
    else:
        ax = None
        ax2 = None
        ax3 = None
        ax4 = None

    # Loop over the waveforms
    peak_values, min_values, max_values = loop_on_waveforms(vtp_values, files, start_waveform, stop_waveform, peaking_time, ax, pt_line, plotting)

    # Compute values at fixed peaking time
    m_PT, b_PT, pvalue_PT, rvalue_PT, fit_end_PT = values_at_peaking_time(ax2, peak_values, peaking_time, fit_start_peak, fit_end_peak, auto_fit, plotting)

    # Compute amplitude at true peak wrt vtp
    m_peak, b_peak, pvalue_peak, rvalue_peak, fit_end_true = amplitude_at_true_peak(ax4, vtp_values, min_values, max_values, start_waveform, stop_waveform, positive_waveforms, auto_fit, fit_start_true, fit_end_true, plotting)

    # Plot for the time at true peak wrt vtp
    ideal_PT = time_at_true_peak(ax3, vtp_values, min_values, max_values, start_waveform, stop_waveform, positive_waveforms, fit_end_true, plotting)

    if plotting:
        if pt_line:
            plt.savefig('plots/vtp_scan_amp_'+file_path.split('/')[-1]+note+'.png', format='png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig('plots/vtp_scan_no_pt_line_amp_'+file_path.split('/')[-1]+note+'.png', format='png', dpi=300, bbox_inches='tight')

    return (m_PT, b_PT, pvalue_PT, rvalue_PT, m_peak, b_peak, pvalue_peak, rvalue_peak, fit_end_PT, ideal_PT)