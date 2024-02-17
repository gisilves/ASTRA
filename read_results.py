import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

if __name__ == '__main__':
    # Check if the correct number of arguments was given
    if len(sys.argv) != 2:
        print("Usage: python read_results.py <file_path>")
        sys.exit(1)

    # Read the data from the file indicated in the command line
    data = np.load(sys.argv[1])

    file_paths = data['file_paths']
    min_y = data['min_y']
    x_at_min = data['x_at_min']
    widths = data['widths']

    # New figure for the 2d histogram
    fig2, axs2 = plt.subplots(3, 1, tight_layout=True, figsize=(12, 10))
    # Find indexes where distance is not -1
    valid_indexes = [i for i, x in enumerate(widths) if x > 1.5e-6]
    # Add the 2d histograms in the same panel    
    num_bins = 100
    
    sns.histplot(x=[widths[i] for i in valid_indexes], y=[min_y[i] for i in valid_indexes], bins=num_bins, ax=axs2[0], cbar=True)
    sns.histplot(x=[widths[i] for i in valid_indexes], y=[x_at_min[i] for i in valid_indexes], bins=num_bins, ax=axs2[1], cbar=True)
    sns.histplot(x=[min_y[i] for i in valid_indexes], y=[x_at_min[i] for i in valid_indexes], bins=num_bins, ax=axs2[2], cbar=True)
    
    # Add titles and labels
    axs2[0].set_title('Waveform width at 80% vs. Minimum V Value')
    axs2[0].set_xlabel('Width')
    axs2[0].set_ylabel('Minimum V Value')
    axs2[1].set_title('Waveform width at 80% vs. Time Value at Minimum V Value')
    axs2[1].set_xlabel('Width')
    axs2[1].set_ylabel('Time Value at Minimum V Value')
    axs2[2].set_title('Minimum V Value vs. Time Value at Minimum V Value')
    axs2[2].set_xlabel('Minimum V Value')
    axs2[2].set_ylabel('Time Value at Minimum V Value')

    # Save the figures
    name = sys.argv[1].split('_')[-1].split('.')[0]
    fig2_name = 'plots/' + name + '_2d_histograms.png'
    #fig2.set_size_inches(16, 9)
    # Add title to the figure with: file name, number of files, number of files after cuts
    fig2.suptitle(name + ' - ' + str(len(file_paths)) + ' files')
    fig2.savefig(fig2_name, dpi=300)

    # Extract data from the plot
    hist_values, x_edges, y_edges = np.histogram2d([widths[i] for i in valid_indexes], [x_at_min[i] for i in valid_indexes], bins=num_bins)

    # Create a 2D NumPy array
    hist_array = np.array(hist_values)

    # Find peaks in the histogram
    # Threshold is 90% of the maximum value
    threshold = 0.9 * np.max(hist_array)
    min_distance = 20 
    peaks, _ = find_peaks(hist_array.flatten(), height=threshold, distance=min_distance)

    # Convert flat indices to 2D indices
    peaks_2d = np.unravel_index(peaks, hist_array.shape)
    # Sort the peaks by the number of counts
    sorted_peaks = sorted(zip(hist_array[peaks_2d], peaks_2d[0], peaks_2d[1]), reverse=True)

    # Select the peak
    peak_idx = 0

    # Cut values set around the peak
    distance_down = x_edges[sorted_peaks[peak_idx][1]] - 1.5 * (x_edges[1] - x_edges[0])
    distance_up = x_edges[sorted_peaks[peak_idx][1]] + 1.5 * (x_edges[1] - x_edges[0])
    x_at_min_down = y_edges[sorted_peaks[peak_idx][2]] - 5 * (y_edges[1] - y_edges[0])
    x_at_min_up = y_edges[sorted_peaks[peak_idx][2]] + 5 * (y_edges[1] - y_edges[0])

    # Add lines superimposed on the plots to indicate the cuts
    axs2[1].axhline(x_at_min_down, color='red')
    axs2[1].axhline(x_at_min_up, color='red')
    axs2[1].axvline(distance_down, color='red')
    axs2[1].axvline(distance_up, color='red')

    # Create a mask of indexes that pass the cuts for distance and x_at_min
    dist_good_idxs = []
    for i, x in enumerate(widths):
        if x != -1:
            if x > distance_down and x < distance_up:
                dist_good_idxs.append(i)

    x_at_min_good_idxs = []
    for i, x in enumerate(x_at_min):
        if x > x_at_min_down and x < x_at_min_up:
            x_at_min_good_idxs.append(i)

    # Create a mask with the common indexes
    common_idxs = list(set(dist_good_idxs) & set(x_at_min_good_idxs))
    mask_array = np.zeros(len(file_paths), dtype=int)
    mask_array[common_idxs] = 1


    # Create the filtered arrays
    filtered_file_paths = file_paths[mask_array == 1]
    filtered_min_y = min_y[mask_array == 1]
    filtered_x_at_min = x_at_min[mask_array == 1]
    filtered_widths = widths[mask_array == 1]

    sns.histplot(x=filtered_widths, y=filtered_min_y, bins=int(num_bins/5), ax=axs2[0], color='red')
    sns.histplot(x=filtered_min_y, y=filtered_x_at_min, bins=int(num_bins/5), ax=axs2[2], color='red')
    
    print("\n")
    print("Number of files:", len(file_paths))
    print("Number of files after cuts:", len(filtered_file_paths))

    # From the filtered_file_paths, extract the measurement number (last 5 digits before .trc), and add it to a list
    measurement_numbers = [int(str.split('.')[0][-5:]) for str in filtered_file_paths]
    print(measurement_numbers)

    # Save the figures
    name = sys.argv[1].split('results_')[1].split('.')[0]
    fig2_name = 'plots/' + name + '_2d_histograms_filtered.png'

    # Add title to the figure with: file name, number of files, number of files after cuts
    fig2.suptitle(name + ' - ' + str(len(file_paths)) + ' files - ' + str(len(filtered_file_paths)) + ' files after cuts')
    fig2.savefig(fig2_name, dpi=300)

    plt.show()

    


