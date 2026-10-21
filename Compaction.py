from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / 'Compaction_Data'
OUTPUT_DIR = BASE_DIR / 'Compaction_output'
TIMESTAMP_UNIT = 'ms'
TRANSITION_TIMES = {}
QUIET_WINDOW_S = 1.0
QUIET_FRACTION = 0.20
MIN_QUIET_POINTS = 8
SMOOTH_SEGMENTS = 21
REFERENCE_SEGMENTS = 5
DPI = 600
NORMALIZATION_METHOD = 'zscore'

COLORS = ('red', 'green', 'blue')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
})

def load_data(path):
    if path.suffix.lower() in ('.xlsx', '.xls'):
        data = pd.read_excel(path)
    else:
        data = pd.read_csv(path)
        if len(data.columns) == 1:
            data = pd.read_csv(path, sep=None, engine='python')
    columns = ['timestamp', 'accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f'{path.name}: missing columns {missing}')
    data = data[columns].apply(pd.to_numeric, errors='raise')
    if data.isna().any().any():
        raise ValueError(f'{path.name}: missing sensor values')
    time = data['timestamp'].to_numpy(dtype=float)
    time = (time - time[0]) / (1000 if TIMESTAMP_UNIT == 'ms' else 1)
    if len(time) < MIN_QUIET_POINTS or np.any(np.diff(time) <= 0):
        raise ValueError(f'{path.name}: invalid timestamps')
    return data, time

def calculate_orientation(data, time):
    acceleration = data[['accX', 'accY', 'accZ']].to_numpy(dtype=float)
    rows = []
    for start in np.arange(0, time[-1], QUIET_WINDOW_S):
        stop = min(start + QUIET_WINDOW_S, time[-1])
        indices = np.flatnonzero((time >= start) &
                                 ((time <= stop) if stop >= time[-1] else (time < stop)))
        if len(indices) < MIN_QUIET_POINTS:
            continue
        values = acceleration[indices]
        width = min(len(values), max(MIN_QUIET_POINTS,
                                    int(np.ceil(QUIET_FRACTION * len(values)))))
        sums = np.vstack((np.zeros((1, 3)), np.cumsum(values, axis=0)))
        squares = np.vstack((np.zeros((1, 3)), np.cumsum(values ** 2, axis=0)))
        means = (sums[width:] - sums[:-width]) / width
        variance = np.maximum((squares[width:] - squares[:-width]) / width - means ** 2, 0)
        offset = int(np.argmin(np.sqrt(variance).sum(axis=1)))
        quiet = values[offset:offset + width]
        median = np.median(quiet, axis=0)
        mad = np.median(np.abs(quiet - median), axis=0)
        scores = 0.6745 * (quiet - median) / np.where(mad > 0, mad, 1)
        selected = np.all(np.abs(scores) <= 3.5, axis=1)
        ax, ay, az = np.mean(quiet[selected] if np.any(selected) else quiet, axis=0)
        roll = np.degrees(np.arctan2(ay, az))
        pitch = np.degrees(np.arctan2(-ax, np.hypot(ay, az)))
        tilt = np.degrees(np.arctan2(np.hypot(ax, ay), az))
        rows.append((0.5 * (start + stop), roll, pitch, tilt))
    if not rows:
        raise ValueError('No suitable quiet intervals found for orientation')
    result = pd.DataFrame(rows, columns=['time', 'Roll', 'Pitch', 'Tilt'])
    result['Roll'] = np.degrees(np.unwrap(np.radians(result['Roll'].to_numpy())))
    for name in ('Roll', 'Pitch', 'Tilt'):
        reference = result[name].iloc[:REFERENCE_SEGMENTS].mean()
        relative = result[name] - reference
        smoothed = (relative.rolling(SMOOTH_SEGMENTS, center=True, min_periods=1).median()
                    .rolling(SMOOTH_SEGMENTS, center=True, min_periods=1).mean())
        result[name] = smoothed - smoothed.iloc[0]
    return result

def add_transition(axis, name, end_time):
    transition = TRANSITION_TIMES.get(name)
    if transition is None or not (0 < transition < end_time):
        return
    axis.axvline(transition, color='0.3', linestyle='--', linewidth=1.1)
    axis.annotate('Tamping compaction', xy=(transition, 0.72),
                  xycoords=('data', 'axes fraction'), xytext=(-9, 0),
                  textcoords='offset points', ha='right', va='center', fontsize=9)
    axis.annotate('Vibratory compaction', xy=(transition, 0.72),
                  xycoords=('data', 'axes fraction'), xytext=(9, 0),
                  textcoords='offset points', ha='left', va='center', fontsize=9)

def normalize_signal(values):
    values = np.asarray(values, dtype=float)
    if NORMALIZATION_METHOD == 'zscore':
        scale = np.std(values)
        return np.zeros_like(values) if scale == 0 else (values - np.mean(values)) / scale
    if NORMALIZATION_METHOD == 'minmax':
        span = np.max(values) - np.min(values)
        return np.zeros_like(values) if span == 0 else 2 * (values - np.min(values)) / span - 1
    raise ValueError('NORMALIZATION_METHOD must be zscore or minmax')

def plot_signals(records, channels, ylabel, filename):
    fig, axes = plt.subplots(len(records), 1, figsize=(11, max(3.1, 2.8 * len(records))),
                             sharex=True, squeeze=False, constrained_layout=True)
    axes = axes[:, 0]
    for axis, (name, data, time) in zip(axes, records):
        for channel, color, direction in zip(channels, COLORS, ('x', 'y', 'z')):
            symbol = 'a' if channels[0].startswith('acc') else r'\omega'
            values = data[channel].to_numpy(dtype=float)
            if channel.startswith('gyro'):
                values = normalize_signal(values)
            axis.plot(time, values, color=color,
                      linewidth=0.7, alpha=0.85, zorder=3 - ('x', 'y', 'z').index(direction),
                      label=rf'${symbol}_{{{direction}}}$')
        axis.set_title(name, pad=5)
        axis.set_ylabel(ylabel)
        axis.set_xlim(left=0)
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.tick_params(direction='in')
        axis.legend(loc='upper right', ncol=3, fontsize=8, framealpha=0.9)
        add_transition(axis, name, time[-1])
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_xlim(0, max(time[-1] for _, _, time in records))
    fig.savefig(OUTPUT_DIR / filename, dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def plot_orientation(records):
    fig, axes = plt.subplots(len(records), 1, figsize=(11, max(3.1, 2.8 * len(records))),
                             sharex=True, squeeze=False, constrained_layout=True)
    axes = axes[:, 0]
    for axis, (name, data, time) in zip(axes, records):
        orientation = calculate_orientation(data, time)
        for angle, color in zip(('Roll', 'Pitch', 'Tilt'), COLORS):
            axis.plot(orientation['time'], orientation[angle], color=color,
                      linewidth=0.9, label=angle)
        axis.set_title(name, pad=5)
        axis.set_ylabel('Relative orientation (°)')
        axis.set_xlim(left=0)
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.tick_params(direction='in')
        if axis is axes[0]:
            axis.legend(loc='upper right', ncol=3, fontsize=8)
        add_transition(axis, name, time[-1])
    axes[-1].set_xlabel('Time (s)')
    axes[-1].set_xlim(0, max(time[-1] for _, _, time in records))
    fig.savefig(OUTPUT_DIR / 'Orientation.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)

def main():
    if not INPUT_PATH.is_dir():
        raise NotADirectoryError(f'Input folder not found: {INPUT_PATH.resolve()}')

    extensions = {'.txt', '.csv', '.xlsx', '.xls'}
    paths = {}
    for path in INPUT_PATH.iterdir():
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        match = re.search(r'layer[_\s-]*([456])(?!\d)', path.stem, re.IGNORECASE)
        if match:
            number = int(match.group(1))
            if number in paths:
                raise ValueError(f'Multiple files found for Layer{number}: '
                                 f'{paths[number].name}, {path.name}')
            paths[number] = path

    missing = [number for number in (4, 5, 6) if number not in paths]
    if missing:
        raise FileNotFoundError(f'Missing data files for layers: {missing}')

    records = [(f'Layer{number}', *load_data(paths[number])) for number in (4, 5, 6)]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_signals(records, ('accX', 'accY', 'accZ'), 'Acceleration (g)', 'Acceleration.png')
    plot_signals(records, ('gyroX', 'gyroY', 'gyroZ'),
                 'Normalized angular velocity (z-score)', 'Angular_Velocity.png')
    plot_orientation(records)
    print(f'Figures saved to {OUTPUT_DIR.resolve()}')

if __name__ == '__main__':
    main()
