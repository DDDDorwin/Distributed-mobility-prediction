from os import listdir
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List


# call the police, I don't care
# this is just for analysis, not meant to be run by everyone
RAW_DIR = "E:/TelecomData/raw/"
PKL_FILE = "activity_per_square.pkl"
SQUARE_ID = "square_id"
TIME_INTERVAL = "time_interval"
COUNTRY_CODE = "country_code"
SMS_IN = "sms_in"
SMS_OUT = "sms_out"
CALL_IN = "call_in"
CALL_OUT = "call_out"
INTERNET = "internet"
ACTIVITY = "activity"


def summarize_activity():
    """
    Summarize all CDRs for every square.
    Discard time and country code.
    Save as new column 'activity'
    """

    # Initialize the main DataFrame with 'squareid' and 'activity'
    df = pd.DataFrame({SQUARE_ID: np.arange(1, 10001), ACTIVITY: np.nan})
    # Set the data types for each column
    df[SQUARE_ID] = df[SQUARE_ID].astype('int64')
    df[ACTIVITY] = df[ACTIVITY].astype('float32')
    
    files = [f for f in listdir(RAW_DIR)]

    for f in tqdm(files, total=len(files), desc="Summarizing Activity"):
        # Read TSV file without headers
        df_file = pd.read_csv(
            join(RAW_DIR, f), header=None, delimiter='\t',
            names=[SQUARE_ID, TIME_INTERVAL, COUNTRY_CODE, SMS_IN, SMS_OUT, CALL_IN, CALL_OUT, INTERNET],
            dtype={SQUARE_ID: 'int64', TIME_INTERVAL: 'int64', COUNTRY_CODE: 'int8',
                    SMS_IN: 'float64', SMS_OUT: 'float64', CALL_IN: 'float64', CALL_OUT: 'float64', INTERNET: 'float64'}
        )

        # Summarize activity for each squareid
        df_summarized = df_file.groupby(SQUARE_ID, as_index=False)[[SMS_IN, SMS_OUT, CALL_IN, CALL_OUT, INTERNET]].sum()
        df_summarized[ACTIVITY] = df_summarized[[SMS_IN, SMS_OUT, CALL_IN, CALL_OUT, INTERNET]].sum(axis=1, skipna=True).astype('float32')

        # Append the summarized DataFrame to the list
        df[ACTIVITY] = df[ACTIVITY].add(df_summarized[ACTIVITY], fill_value=0)

    print(df.head(10))
    df.to_pickle(PKL_FILE, compression=None)


def plot_distribution():
    """Make histogram of activity distribution"""
    
    # Assuming df is your DataFrame with columns sid and activity
    df = pd.read_pickle(PKL_FILE, compression=None)

    # Plot histogram with a logarithmic y-axis scale
    plt.hist(df[ACTIVITY], bins=50, edgecolor='black')
    plt.yscale('log')  # Set y-axis scale to logarithmic

    # Add labels and title
    plt.xlabel('Sum of all CDRs per Square', fontsize=22)
    plt.ylabel('Frequency (log scale)', fontsize=22)
    plt.title('Histogram of sum of CDRs per Square', fontsize=22)

    # Show the plot
    plt.show()


def pick_medians(n=100):
    """Select n squares by the medians of activity"""

    df = pd.read_pickle(PKL_FILE, compression=None)

    # Get sid with the lowest and highest activity
    sid_lowest_activity = df.loc[df[ACTIVITY].idxmin(), SQUARE_ID]
    sid_highest_activity = df.loc[df[ACTIVITY].idxmax(), SQUARE_ID]
    print("SID with the lowest activity:", sid_lowest_activity)
    print("SID with the highest activity:", sid_highest_activity)

    # Calculate the percentiles corresponding to the 100 medians
    percentiles = np.linspace(0, 100, 100)
    median_activity_values = np.percentile(df[ACTIVITY], percentiles)

    # Find the closest SQUARE_ID values to each calculated percentile
    selected_sids = []
    for median_value in median_activity_values:
        closest_sid = df.loc[(df[ACTIVITY] - median_value).abs().idxmin(), SQUARE_ID]
        selected_sids.append(closest_sid)

    # Display the corresponding SQUARE_ID values for the medians
    print("SID values corresponding to the medians:")
    print(selected_sids)
    return selected_sids

# from pick_medians(n=100)
# [2801, 3204, 290, 1432, 1328, 501, 9906, 2327, 1765, 4202, 6025, 498, 930, 2120, 5803, 406, 672, 5385, 4307, 3809, 5889, 1391, 1859, 2725, 5826, 6512, 265, 6134, 2406, 2874, 8426, 2420, 4291, 2840, 9514, 9826, 5885, 9519, 551, 3491, 8299, 6183, 8703, 5527, 4234, 3782, 6580, 3391, 1851, 8701, 8815, 7923, 2253, 6639, 8594, 8699, 7606, 3327, 6216, 7181, 2641, 9390, 9592, 2897, 8231, 7373, 4081, 2689, 8252, 6492, 9597, 8309, 8816, 7992, 8343, 4665, 9759, 2439, 7224, 9561, 7355, 8970, 9659, 8654, 9974, 5747, 7983, 4767, 6077, 8375, 6256, 6960, 8269, 3863, 6047, 4551, 4157, 5048, 5860, 5161]


def pick_linear(n: int = 100) -> List[int]:
    """Select n squares by the linear distribution of activity"""

    # Assuming df is your DataFrame with columns squareid and activity
    df: pd.DataFrame = pd.read_pickle(PKL_FILE, compression=None)

    # Get sid with the lowest and highest activity
    sid_lowest_activity = df.loc[df[ACTIVITY].idxmin(), SQUARE_ID]
    sid_highest_activity = df.loc[df[ACTIVITY].idxmax(), SQUARE_ID]
    print("SID with the lowest activity:", sid_lowest_activity)
    print("SID with the highest activity:", sid_highest_activity)

    # Generate 100 linearly spaced activity values between min and max
    activity_values = np.linspace(df[ACTIVITY].min(), df[ACTIVITY].max(), 100)

    # Find the closest SQUARE_ID values to each calculated activity value
    selected_sids = set()  # Use a set to ensure uniqueness

    for activity_value in activity_values:
        closest_sid = df.loc[(df[ACTIVITY] - activity_value).abs().idxmin(), SQUARE_ID]
        selected_sids.add(closest_sid)
        df = df[df[SQUARE_ID] != closest_sid]

    # Convert the set to a list for consistent output
    selected_sids_list = list(selected_sids)

    # Display the corresponding SQUARE_ID values for the linearly spaced activity values
    print("SID values corresponding to linearly spaced activity values:")
    print(selected_sids_list)
    print("#squares:", len(selected_sids_list))
    
    return selected_sids_list

# from pick_linear(n=100)
# [2801, 1768, 3688, 5897, 3423, 2732, 7850, 8278, 9072, 6955, 6544, 4253, 5848, 7567, 6276, 3659, 5873, 6047, 4863, 7162, 5652, 5457, 3956, 5661, 5249, 5356, 4750, 4647, 5347, 6262, 6059, 5053, 5659, 5251, 6260, 5860, 4648, 4860, 5967, 5068, 6165, 4758, 4456, 4854, 4957, 5963, 6058, 4959, 6062, 6065, 6065, 5855, 5257, 4755, 5955, 4857, 5163, 6064, 4459, 5262, 5261, 5259, 5259, 4856, 4856, 4855, 4855, 4855, 4855, 5159, 5159, 5159, 5159, 5258, 5258, 5258, 5061, 5061, 5061, 5061, 5061, 5061, 5061, 5061, 5061, 5059, 5059, 5059, 5059, 5059, 5059, 5059, 5059, 5059, 5059, 5059, 5161, 5161, 5161, 5161]


def pick_location(n = 100):
    grid_size = 100
    # Calculate the number of subgrids along each axis
    subgrid_size = int(np.sqrt(n))
    step_size = grid_size // subgrid_size

    # Initialize the list to store selected square IDs
    selected_squares = []

    # Loop through each subgrid and select one square
    for i in range(subgrid_size):
        for j in range(subgrid_size):
            # Calculate the square ID based on grid position
            square_id = i * step_size * grid_size + j * step_size + 1
            selected_squares.append(square_id)
    
    print("SID values corresponding to physically spaced activity values:")
    print(selected_squares)
    print("#squares:", len(selected_squares))

    return selected_squares

# from pick_location(n=100):
# [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 1001, 1011, 1021, 1031, 1041, 1051, 1061, 1071, 1081, 1091, 2001, 2011, 2021, 2031, 2041, 2051, 2061, 2071, 2081, 2091, 3001, 3011, 3021, 3031, 3041, 3051, 3061, 3071, 3081, 3091, 4001, 4011, 4021, 4031, 4041, 4051, 4061, 4071, 4081, 4091, 5001, 5011, 5021, 5031, 5041, 5051, 5061, 5071, 5081, 5091, 6001, 6011, 6021, 6031, 6041, 6051, 6061, 6071, 6081, 6091, 7001, 7011, 7021, 7031, 7041, 7051, 7061, 7071, 7081, 7091, 8001, 8011, 8021, 8031, 8041, 8051, 8061, 8071, 8081, 8091, 9001, 9011, 9021, 9031, 9041, 9051, 9061, 9071, 9081, 9091]


def plot_selection(selection: List[int], method: str):
    """Plot the activity of some selected squares"""

    df = pd.read_pickle(PKL_FILE, compression=None)

    # Assuming df is your DataFrame with columns sid and activity
    # and selection is the list of sid values you want to retain

    # Retain rows with the specified sid values
    selected_rows = df[df[SQUARE_ID].isin(selection)]

    # Sort the selected rows by ACTIVITY in descending order
    selected_rows = selected_rows.sort_values(by=ACTIVITY, ascending=False)

    # Cast SQUARE_ID to string
    selected_rows[SQUARE_ID] = selected_rows[SQUARE_ID].astype(str)

    # Plot the ACTIVITY for the selected rows with SQUARE_ID as the x-axis
    plt.bar(selected_rows[SQUARE_ID], selected_rows[ACTIVITY])
    plt.xlabel('Selected Squares (too crowded)', fontsize=22)
    plt.ylabel('Sum of CDRs per Square', fontsize=22)
    # plt.title('Sum of CDRs for Squares selected as medians', fontsize=22)
    plt.title(f'Sum of CDRs for Squares {}', fontsize=22)
    plt.xticks([])
    plt.show()


if __name__ == "__main__":
    summarize_activity()
    plot_distribution()
    median_selections = pick_medians()
    plot_selection(median_selections, "selected by medians")
    linear_selections = pick_linear()
    plot_selection(linear_selections, "selected by linear distribution")
    location_selections = pick_location()
    plot_selection(location_selections, "selected by physical location")
