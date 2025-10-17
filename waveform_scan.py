import sys
import glob
import numpy as np
import concurrent.futures
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from lecroyutils.data import LecroyScopeData
import os

def process_waveform(file_path, do_plotting=False, verbose=False):
    # Load data from the file
    data = LecroyScopeData.parse_file(file_path)

    if verbose:
        print("Data loaded: ", file_path)
        print("\tNumber of points: ", len(data.x))
    
    # Get x and y values from the loaded data
    x_values = data.x
    y_values = data.y

    # Convert x_values to us (total width of the waveform is 100 us)
    x_values = x_values * 1e6

    # Remove the last 10% of the data to avoid exclude the positive peak
    x_values = x_values[:-int(len(x_values) * 0.1)]
    y_values = y_values[:-int(len(y_values) * 0.1)]

    if verbose:
        print("\tNumber of points after removing the last 10%: ", len(x_values))

    # Compute y data variance
    y_variance = np.var(y_values)
    if verbose:
        print("\tVariance of y values: ", y_variance)

    # Smooth the y_values to reduce noise (adjust the smoothing parameter as needed)
    smoothed_y_values = UnivariateSpline(x_values, y_values, s=2)(x_values)

    # Calculate the baseline using the mean or median of the initial portion of the smoothed data
    num_points_waveform = len(x_values)
    baseline_window = num_points_waveform // 12
    baseline = np.median(smoothed_y_values[:baseline_window])
    
    # Find the index of the absolute minimum in the smoothed y_values
    absolute_min_index = np.argmin(smoothed_y_values)
    absolute_min_y = smoothed_y_values[absolute_min_index]

    # Find the index of the absolute maximum in the smoothed y_values, before the absolute minimum
    if absolute_min_index == 0:
        print("Absolute minimum is at the beginning of the waveform")
        print("Skipping file:", file_path)
        
        return [file_path,
                (0, 0),
                (0, 0),
                0,
                [(0, 0)],
                0,
                0,
                0,
                y_variance,
                False]
    
    absolute_max_index = np.argmax(smoothed_y_values[:absolute_min_index+1])
    absolute_max_y = smoothed_y_values[absolute_max_index]

    # Compute the overshoot of the waveform (percentage of the absolute maximum over the absolute minimum)
    overshoot = 100 * (absolute_max_y - baseline) / (baseline - absolute_min_y)

    # Find indices where smoothed_y_values cross the threshold
    cross_threshold_indices = np.where(np.diff(np.sign(smoothed_y_values - (absolute_min_y - (0.2 * (absolute_min_y - baseline))))))[0]

    # Use the midpoint of each crossing interval as the intersection point
    intersection_points = [(x_values[i] + x_values[i + 1]) / 2 for i in cross_threshold_indices]

    # If we have more than 2 intersection points, keep only the one closest to the absolute minimum from the left and right
    if len(intersection_points) > 2:
        left_intersection_index = np.argmin(np.abs(intersection_points - x_values[absolute_min_index]))
        right_intersection_index = np.argmin(np.abs(intersection_points - x_values[absolute_min_index]))
        intersection_points = [intersection_points[left_intersection_index], intersection_points[right_intersection_index]]

    if do_plotting:
        # Use seaborn for plotting
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Plot the original data, smoothed data, absolute minimum, absolute maximum, baseline, and selected points
        plt.plot(x_values, y_values, label='Original Data')
        plt.plot(x_values, smoothed_y_values, label='Smoothed Data', linestyle='dashed')
        plt.scatter(x_values[absolute_min_index], absolute_min_y, color='red', label='Absolute Minimum')
        plt.scatter(x_values[absolute_max_index], absolute_max_y, color='green', label='Absolute Maximum')
        plt.scatter(intersection_points, smoothed_y_values[cross_threshold_indices], color='blue', label='Intersection Points with Threshold')
        plt.axhline(y=baseline, color='blue', linestyle='--', label='Baseline')
        plt.axhline(y=absolute_min_y, color='orange', linestyle='--', label='Absolute Minimum')
        plt.axhline(y=absolute_min_y - (0.2 * (absolute_min_y - baseline)), color='purple', linestyle='--', label='Threshold')
        plt.axhline(y=absolute_max_y, color='green', linestyle='--', label='Absolute Maximum')

        # Mark the last point used for baseline
        plt.scatter(x_values[baseline_window - 1], smoothed_y_values[baseline_window - 1], color='black', marker='x', label='Last Point for Baseline')
        
        # Add title, labels, and legend
        plt.xlabel('X values (us)')
        plt.ylabel('Y values (V)')

        # Check if we are on Windows
        if os.name == 'nt':
            plt.title(file_path.split('\\')[-1]) 
        else:
            plt.title(file_path.split('/')[-1])
        plt.legend()
        plt.show()

    # Calculate the width between the two intersection points
    if len(intersection_points) < 2:
        width = -1
    else:
        width = np.abs(intersection_points[1] - intersection_points[0])

    if verbose:
        print("\tAbsolute Minimum at (x, y):", (x_values[absolute_min_index], absolute_min_y))
        print("\tAbsolute Maximum at (x, y):", (x_values[absolute_max_index], absolute_max_y))
        print("\tOvershoot (%):", "{:.2f}".format(overshoot))
        print("\tThreshold Value:", absolute_min_y - (0.2 * (absolute_min_y - baseline)))
        print("\tBaseline Value:", baseline)
        print("\tIntersection Points with Threshold at (x, y):\n\t", [(x, smoothed_y_values[np.abs(x_values - x).argmin()]) for x in intersection_points])
        print("\tWidth between Intersection Points:", width)
        
    # Return a list of the results
    return [file_path, 
            (x_values[absolute_min_index], absolute_min_y), 
            (x_values[absolute_max_index], absolute_max_y), 
            overshoot, 
            [(x, smoothed_y_values[np.abs(x_values - x).argmin()]) for x in intersection_points], 
            width, 
            absolute_min_y - (0.2 * (absolute_min_y - baseline)), 
            baseline,
            y_variance,
            True]

def process_file(file):
    results = process_waveform(file, do_plotting=False, verbose=False)
    return results

if __name__ == '__main__':
    # Check if a file path and thread count are provided as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python script.py <file_path> <num_threads>")
        sys.exit(1)

    # Extract the file path and number of threads from the command-line arguments
    file_path = sys.argv[1]
    num_threads = int(sys.argv[2])

    # Check if the number of threads is valid
    if num_threads < 1:
        print("Number of threads must be greater than 0")
        sys.exit(1)

    # Check if the file path is valid
    if not glob.glob(file_path):
        print("Invalid file path")
        sys.exit(1)

    # Find all the .trc files in the given path
    files = glob.glob(file_path + '/*.trc')
    # Sort the files by name
    files.sort()

    # Lists for results
    min_y = []
    max_y = []
    x_at_min = []
    widths = []
    overshoot = []
    y_variances = []
    baselines = []

    # Use ThreadPoolExecutor for parallel processing with specified number of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Use tqdm for progress bar
        results_list = list(tqdm(executor.map(process_file, files), total=len(files), desc="Processing Files"))



    # Collect results (filter out the results that are not valid)
    results_list = [result for result in results_list if result[-1] == True]

    file_paths = [result[0] for result in results_list]    
    
    min_y = [result[1][1] for result in results_list]
    x_at_min = [result[1][0] for result in results_list]
    
    max_y = [result[2][1] for result in results_list]
    
    overshoot = [result[3] for result in results_list]   
    
    widths = [result[5] for result in results_list]
    
    widths_filtered = [result[5] for result in results_list if result[5] != -1]

    baselines = [result[7] for result in results_list]

    y_variances = [result[8] for result in results_list]

    # Save the lists to file
    out_name = 'Results/results_' + file_path.split('/')[-1]
    np.savez_compressed(out_name, 
                        file_paths=file_paths, 
                        min_y=min_y, 
                        x_at_min=x_at_min,
                        max_y=max_y, 
                        widths=widths,
                        overshoot=overshoot)
    
    num_bins = 100
    fig, axs = plt.subplots(3, 1, tight_layout=True)
    # Plot the histograms in the same panel
    axs[0].hist(widths_filtered, bins=num_bins)
    axs[1].hist(min_y, bins=num_bins)
    axs[2].hist(x_at_min, bins=num_bins)

    # Add titles and labels
    axs[0].set_title('Distance between Intersection Points')
    axs[0].set_xlabel('Distance')
    axs[0].set_ylabel('Count')
    axs[1].set_title('Minimum Y Value')
    axs[1].set_xlabel('Minimum Y Value')
    axs[1].set_ylabel('Count')
    axs[2].set_title('X Value at Minimum Y Value')
    axs[2].set_xlabel('X Value at Minimum Y Value')
    axs[2].set_ylabel('Count')

    # New figure for the 2d histogram
    fig2, axs2 = plt.subplots(3, 1, tight_layout=True, figsize=(12, 10))
    # Find indexes where distance is not -1
    valid_indexes = [i for i, x in enumerate(widths) if x != -1]
    # Add the 2d histograms in the same panel    
    sns.histplot(x=[widths[i] for i in valid_indexes], y=[min_y[i] for i in valid_indexes], bins=num_bins, ax=axs2[0], cbar=True)
    sns.histplot(x=[widths[i] for i in valid_indexes], y=[x_at_min[i] for i in valid_indexes], bins=num_bins, ax=axs2[1], cbar=True)
    sns.histplot(x=[min_y[i] for i in valid_indexes], y=[x_at_min[i] for i in valid_indexes], bins=num_bins, ax=axs2[2], cbar=True)

    # Add titles and labels
    axs2[0].set_title('Distance between Intersection Points vs. Minimum Y Value')
    axs2[0].set_xlabel('Distance')
    axs2[0].set_ylabel('Minimum Y Value')
    axs2[1].set_title('Distance between Intersection Points vs. X Value at Minimum Y Value')
    axs2[1].set_xlabel('Distance')
    axs2[1].set_ylabel('X Value at Minimum Y Value')
    axs2[2].set_title('Minimum Y Value vs. X Value at Minimum Y Value')
    axs2[2].set_xlabel('Minimum Y Value')
    axs2[2].set_ylabel('X Value at Minimum Y Value')
    


    # Save the figures
    # Retrieve the folder name from the file path
    folder_name = 'plots/' + file_path.split('/')[-1]
    fig1_name = folder_name + '_histograms.png'
    fig2_name = folder_name + '_2d_histograms.png'
    # Save figure at high resolution
    fig.savefig(fig1_name, dpi=300)
    fig2.savefig(fig2_name, dpi=300)