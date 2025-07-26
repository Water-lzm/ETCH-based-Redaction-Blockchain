import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load Data ---
excel_file_path = r'C:\Users\lzm\Desktop\all data with four interval.xlsx'
sheet_name = 'Sheet1'
header_row = 0

def load_data_from_excel(file_path, sheet, header):
    print(f"Reading data from file '{file_path}', sheet '{sheet}'...")

    data_blocks_cols = {
        720: 'A:E',
        840: 'G:K',
        960: 'M:Q',
        1000: 'S:W'
    }

    standard_columns = [
        'Time',
        'Regular_ChainLength',
        'Regular_Rebuild_ms',
        'ETCH_ChainLength',
        'ETCH_Edit_ms'
    ]

    datasets = {}
    try:
        for interval, cols in data_blocks_cols.items():
            df = pd.read_excel(
                io=file_path,
                sheet_name=sheet,
                header=header,
                usecols=cols
            )

            df.dropna(how='all', inplace=True)
            df.columns = standard_columns
            df = df.astype(float)

            datasets[interval] = df
            print(f"Loaded data for interval {interval}s, total {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error while reading Excel: {e}")
        return None

    return datasets

# --- 2. Plotting Function (Separate Rows) ---
def plot_dashboard(datasets):
    if not datasets:
        print("No data loaded, cannot generate charts.")
        return

    plt.rcParams.update({
        'font.size': 18,
        'font.family': 'serif',
        'axes.labelsize': 15,
        'axes.titlesize': 18,
        'legend.fontsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'lines.linewidth': 4,
        'lines.markersize': 6
    })

    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

    colors = {
        'regular': 'tab:blue',
        'etch': 'tab:orange',
        'saving': 'tab:green'
    }

    for interval, df in datasets.items():
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        #fig.suptitle(f'{interval}s Interval - ETCH vs General Blockchain', fontsize=14, weight='bold')

        mask_better = df['Regular_ChainLength'] > df['ETCH_ChainLength']

        # --- Column 0: Chain Length ---
        ax = axes[0]
        ax.plot(df['Time'][mask_better], df['Regular_ChainLength'][mask_better],
                linestyle='-', color=colors['regular'], label='General BC')
        ax.plot(df['Time'][mask_better], df['ETCH_ChainLength'][mask_better],
                linestyle='--', color=colors['etch'], label='ETCH')
        ax.set_ylabel('Chain Length (MB)')
        ax.set_xlabel('Elapsed Time (s)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        # --- Column 1: Update Overhead ---
        ax = axes[1]
        ax.plot(df['Time'], df['Regular_Rebuild_ms'].cumsum() / 1000,
                linestyle='-', color=colors['regular'], label='General BC')
        ax.plot(df['Time'], df['ETCH_Edit_ms'].cumsum() / 1000,
                linestyle='--', color=colors['etch'], label='ETCH')
        ax.set_xlabel('Elapsed Time (s)')
        ax.set_ylabel('Update Overhead (s)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        # --- Column 2: Saving Rate ---
        ax = axes[2]
        savings_raw = (df['Regular_ChainLength'] - df['ETCH_ChainLength']) / df['Regular_ChainLength'].replace(0, np.nan)
        savings = 100 * savings_raw
        savings = savings.where(savings > 0)
        print(f'Average saving ratio for {interval}s: {savings_raw.mean():.4f}')
        ax.plot(df['Time'], savings.fillna(0), linestyle='-', color=colors['saving'])
        ax.set_xlabel('Elapsed Time (s)')
        ax.set_ylabel('Saving Rate (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space for suptitle
        plt.show()

# --- Run ---
all_datasets = load_data_from_excel(excel_file_path, sheet_name, header_row)
plot_dashboard(all_datasets)
