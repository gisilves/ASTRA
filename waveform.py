import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from lecroyutils.data import LecroyScopeData

# Use seaborn for plotting
sns.set(style="whitegrid")
# Use bigger fonts
sns.set_context("talk", font_scale=1.5)

def process_waveform(file_path, do_plotting=False, verbose=False):
    # Load data from the file
    data = LecroyScopeData.parse_file(file_path)
    print("Data loaded: ", file_path)
    print("\tNumber of points: ", len(data.x))
    
    # Get x and y values from the loaded data
    x_values = data.x
    y_values = data.y

    # Convert x_values to seconds (total width of the waveform is 100 us)
    x_values = x_values * 1e6

    # Remove the last 10% of the data to avoid exclude the positive peak
    x_values = x_values[:-int(len(x_values) * 0.1)]
    y_values = y_values[:-int(len(y_values) * 0.1)]

    print("\tNumber of points after removing the last 10%: ", len(x_values))
    # Compute y data variance
    y_variance = np.var(y_values)
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
    absolute_max_index = np.argmax(smoothed_y_values[:absolute_min_index+1])
    absolute_max_y = smoothed_y_values[absolute_max_index]

    # Compute the overshoot of the waveform (percentage of the absolute maximum over the absolute minimum)
    overshoot = np.abs(100 * (absolute_max_y - baseline) / (baseline - absolute_min_y))

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

        # Create a new figure
        plt.figure(figsize=(10, 6))
        # Plot the original data
        plt.plot(x_values, y_values, label='Acquired Data', linewidth=1.5)
        plt.xlabel('X values (us)', fontsize=18)
        plt.ylabel('Y values (V)', fontsize=18)
        # Check if we are on Windows
        if os.name == 'nt':
            plt.title(file_path.split('\\')[-1])
        else:
            plt.title(file_path.split('/')[-1])
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        # Plot the original data, smoothed data, absolute minimum, absolute maximum, baseline, and selected points
        plt.plot(x_values, y_values, label='Original Data')
        plt.plot(x_values, smoothed_y_values, label='Smoothed Data', linestyle='dashed')
        # Add shaded area around the baseline, with height equal to y_variance
        plt.fill_between(x_values, baseline - y_variance, baseline + y_variance, color='gray', alpha=0.2, label='Baseline +/- Variance')

        plt.scatter(x_values[absolute_min_index], absolute_min_y, color='red', label='Absolute Minimum')
        plt.scatter(x_values[absolute_max_index], absolute_max_y, color='green', label='Absolute Maximum')
        plt.scatter(intersection_points, smoothed_y_values[cross_threshold_indices], color='blue', label='Intersection Points with Threshold')
        plt.axhline(y=baseline, color='blue', linestyle='--', label='Baseline')
        plt.axhline(y=absolute_min_y, color='orange', linestyle='--', label='Absolute Minimum')
        plt.axhline(y=absolute_min_y - (0.2 * (absolute_min_y - baseline)), color='purple', linestyle='--', label='Threshold')
        plt.axhline(y=absolute_max_y, color='green', linestyle='--', label='Absolute Maximum')

        # Mark the last point used for baseline
        plt.scatter(x_values[baseline_window - 1], smoothed_y_values[baseline_window - 1], color='black', marker='x', label='Last Point for Baseline')

        plt.xlabel('X values (us)')
        plt.ylabel('Y values (V)')
        # Add title from the file name (only the last part of the path)
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
    # Check if a file path is provided as command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python waveform.py <file_path>")
        sys.exit(1)

    # Get the file path from the command-line arguments
    file_path = sys.argv[1]

    # Process the waveform file
    results = process_waveform(file_path, do_plotting=True, verbose=True)