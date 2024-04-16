import sys
import argparse
import functions

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot Vtp scan')

    parser.add_argument('--file_path', type=str, help='Path to the folder containing the .trc files', default='/media/DATA/', dest='file_path')
    parser.add_argument('--start_waveform', type=int, help='Start waveform', default=0, dest='start_waveform')
    parser.add_argument('--stop_waveform', type=int, help='Stop waveform', default=99, dest='stop_waveform')
    parser.add_argument('--peaking_time', type=float, help='Nominal peaking time of the setup', default=9, dest='peak_time')
    parser.add_argument('--fit_start_peak', type=float, help='Minimum value for the linear fit', default=20, dest='fit_start_peak')
    parser.add_argument('--fit_end_peak', type=float, help='Maximum value for the linear fit', default=140, dest='fit_end_peak')
    parser.add_argument('--fit_start_min', type=float, help='Minimum value for the linear fit', default=20, dest='fit_start_min')
    parser.add_argument('--fit_end_min', type=float, help='Maximum value for the linear fit', default=140, dest='fit_end_min')
    parser.add_argument('--auto_fit', action='store_true', help='Automatically find the fit range', default=False, dest='auto_fit')
    parser.add_argument('--pt_line', action='store_true', help='Add a line at the peaking time', default=False, dest='pt_line')
    parser.add_argument('--note', type=str, help='Note to add to the plot and filename', default='', dest='note')

    # If no arguments are given, print the help message
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    print('Processing Vtp scan with following parameters:')
    print('\tFile path: ' + args.file_path)
    print('\tStart waveform: ' + str(args.start_waveform))
    print('\tStop waveform: ' + str(args.stop_waveform))
    print('\tPeaking time: ' + str(args.peak_time))
    print('\tFit min at peak: ' + str(args.fit_start_peak))
    print('\tFit max at peak: ' + str(args.fit_end_peak))
    print('\tFit min at minimum: ' + str(args.fit_start_min))
    print('\tFit max at minimum: ' + str(args.fit_end_min))
    print('\tAuto fit: ' + str(args.auto_fit))
    print('\tAdd line at peaking time: ' + str(args.pt_line))
    print('\tNote: ' + args.note)
    
    functions.process_folder(args.file_path, args.start_waveform, args.stop_waveform, args.peak_time, args.fit_start_peak, args.fit_end_peak, args.fit_start_min, args.fit_end_min, args.auto_fit, args.pt_line, args.note)