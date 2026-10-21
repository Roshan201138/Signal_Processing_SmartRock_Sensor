from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt, periodogram

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 11, "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
                     "axes.linewidth": 1.0, "savefig.dpi": 600})

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / 'Cyclic_loading_data' / 'Cyclic_loading.txt'
OUTPUT_DIR = BASE_DIR / 'Cyclic_loading_output'
TIMESTAMP_UNIT = 'ms'
LOADING_FREQUENCY_HZ = 2.0
LOWPASS_CUTOFF_HZ = None
SPECTRAL_ENERGY_FRACTION = 0.99
MIN_CUTOFF_HZ = 0.5
MAX_CUTOFF_NYQUIST_FRACTION = 0.8
FILTER_ORDER = 4
HAMPEL_WINDOW_S = 0.5
HAMPEL_THRESHOLD = 6.0
FIG_DPI = 600
ACCELERATION_UNIT = 'g'
GYROSCOPE_UNIT = 'deg/s'
QUIET_SEGMENT_ORIENTATION = True
QUIET_WINDOW_S = 1.0
QUIET_FRACTION = 0.20
SMOOTH_SEGMENTS = 21
MAX_PLOT_POINTS = 100000

RESAMPLE_IRREGULAR_TIMESTAMPS = True
MAX_GAP_FACTOR = 10.0
CHANNEL_ALIASES = {
    'accX': ['accX', 'acc_x', 'ax'],
    'accY': ['accY', 'acc_y', 'ay'],
    'accZ': ['accZ', 'acc_z', 'az'],
    'gyroX': ['gyroX', 'gyro_x', 'gx', 'gyrX', 'angularVelocityX'],
    'gyroY': ['gyroY', 'gyro_y', 'gy', 'gyrY', 'angularVelocityY'],
    'gyroZ': ['gyroZ', 'gyro_z', 'gz', 'gyrZ', 'angularVelocityZ'],
}
ACC = ['accX', 'accY', 'accZ']
GYRO = ['gyroX', 'gyroY', 'gyroZ']
CHANNELS = ACC + GYRO

def load_data(path):
    if not path.is_file():
        raise FileNotFoundError(f'Input file not found: {path.resolve()}')
    if path.suffix.lower() in ('.csv', '.txt'):
        try:
            df = pd.read_csv(path)
            if len(df.columns) == 1:
                df = pd.read_csv(path, sep=None, engine='python')
        except Exception:
            df = pd.read_csv(path, sep=None, engine='python')
    elif path.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(path)
    else:
        raise ValueError('Input must be TXT, CSV, XLSX or XLS')
    columns = {str(c).strip().lower(): c for c in df.columns}
    mapping = {}
    for canonical, aliases in {'timestamp': ['timestamp', 'time', 'time_ms', 'time_s'], **CHANNEL_ALIASES}.items():
        original = next((columns[a.lower()] for a in aliases if a.lower() in columns), None)
        if original is None:
            raise ValueError(f'Missing {canonical}; available columns: {list(df.columns)}')
        mapping[original] = canonical
    df = df.rename(columns=mapping)[['timestamp'] + CHANNELS].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['timestamp']).sort_values('timestamp').drop_duplicates('timestamp')
    if len(df) < 10:
        raise ValueError('Not enough timestamped samples')
    if df[CHANNELS].isna().any().any():
        print('Warning: missing/non-numeric sensor samples interpolated before processing.')
        df[CHANNELS] = df[CHANNELS].interpolate(method='linear', limit_direction='both')
    if df[CHANNELS].isna().any().any():
        raise ValueError('A sensor channel contains no usable numeric measurements')
    t = (df['timestamp'].to_numpy(dtype=float) - float(df['timestamp'].iloc[0]))
    if TIMESTAMP_UNIT.lower() == 'ms':
        t /= 1000.0
    elif TIMESTAMP_UNIT.lower() != 's':
        raise ValueError("TIMESTAMP_UNIT must be 'ms' or 's'")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError('Timestamps must be strictly increasing')
    median_dt = float(np.median(dt))
    fs = 1 / median_dt
    jitter = float(np.max(np.abs(dt - median_dt)) / median_dt)
    original_count = len(df)
    max_gap_factor = float(np.max(dt) / median_dt)
    if jitter > 0.05:
        if not RESAMPLE_IRREGULAR_TIMESTAMPS:
            raise ValueError(f'Irregular timestamps (max interval deviation {jitter:.1%}). '
                             'Enable RESAMPLE_IRREGULAR_TIMESTAMPS.')
        if max_gap_factor > MAX_GAP_FACTOR:
            print(f'WARNING: timestamp gap {max_gap_factor:.1f}x median interval. '
                  'Interpolation cannot recover missing physical events; inspect the record.')
        intervals = max(1, int(round((t[-1] - t[0]) / median_dt)))
        t_uniform = np.linspace(t[0], t[-1], intervals + 1)
        resampled = {'timestamp': float(df['timestamp'].iloc[0]) +
                     t_uniform * (1000.0 if TIMESTAMP_UNIT.lower() == 'ms' else 1.0)}
        for c in CHANNELS:
            resampled[c] = np.interp(t_uniform, t, df[c].to_numpy(dtype=float))
        df = pd.DataFrame(resampled)
        t = t_uniform
        fs = 1.0 / (t[1] - t[0])
        print(f'Resampled irregular timestamps: {original_count} original samples -> '
              f'{len(df)} uniformly spaced samples at {fs:.3f} Hz '
              f'(max interval deviation {jitter:.1%}).')
    df.attrs['original_samples'] = original_count
    df.attrs['timestamp_jitter'] = jitter
    df.attrs['resampled'] = jitter > 0.05
    df.attrs['max_gap_factor'] = max_gap_factor
    return df.reset_index(drop=True), t, fs

def hampel_interpolate(x, t, fs):
    window = max(3, int(round(HAMPEL_WINDOW_S * fs)))
    if window % 2 == 0:
        window += 1
    if window > len(x):
        raise ValueError(f'Hampel window ({window} samples) exceeds record length ({len(x)})')
    local_median = median_filter(x, size=window, mode='nearest')
    local_mad = median_filter(np.abs(x - local_median), size=window, mode='nearest')
    scaled_mad = 1.4826 * local_mad
    flagged = (scaled_mad > 0) & (np.abs(x - local_median) > HAMPEL_THRESHOLD * scaled_mad)
    valid = ~flagged
    if np.sum(valid) < 2:
        raise ValueError('Too few valid samples after Hampel detection')
    replaced = x.copy()
    replaced[flagged] = np.interp(t[flagged], t[valid], x[valid])
    return replaced, flagged, window

def quadratic_detrend(x, t):
    u = 2 * (t - t[0]) / (t[-1] - t[0]) - 1
    trend = np.polyval(np.polyfit(u, x, deg=2), u)
    return x - trend

def export_figure(fig, stem, outdir):
    fig.savefig(outdir / f'{stem}.png', dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)

def plot_raw_filtered(t, p, channels, unit, stem, outdir, x_mode):
    x = t if x_mode == 'Time' else p['loading_cycles'].to_numpy()
    xlabel = 'Time (s)' if x_mode == 'Time' else 'Loading cycles, N (–)'
    fig, axs = plt.subplots(3, 2, figsize=(10.5, 6.7), sharex=True,
                            constrained_layout=True)
    colors = ['#c93333', '#268a50', '#2454bd']
    stride = max(1, len(t) // MAX_PLOT_POINTS)
    symbol = 'a' if channels == ACC else r'\omega'
    for i, (name, color) in enumerate(zip(channels, colors)):
        label = rf'${symbol}_{{{name[-1].lower()}}}$ ({unit})'
        for j, col in enumerate(('raw', 'filtered')):
            axs[i, j].plot(x[::stride], p[f'{name}_{col}'].to_numpy()[::stride],
                           color=color, linewidth=0.65)
            axs[i, j].set_ylabel(label)
            axs[i, j].grid(alpha=0.2, linewidth=0.4)
            axs[i, j].tick_params(direction='in')
    quantity = 'acceleration' if channels == ACC else 'angular velocity'
    axs[0, 0].set_title('(a) Raw ' + quantity)
    axs[0, 1].set_title('(b) Filtered ' + quantity)
    for j in range(2):
        axs[2, j].set_xlabel(xlabel)
    export_figure(fig, f'{stem}_vs_{x_mode}', outdir)

def quiet_orientation(t, p):
    acc = np.column_stack([p[c + '_despiked'].to_numpy() for c in ACC])
    rows = []
    for start in np.arange(t[0], t[-1], QUIET_WINDOW_S):
        ix = np.flatnonzero((t >= start) & (t < start + QUIET_WINDOW_S))
        if len(ix) < 8:
            continue
        n = min(len(ix), max(8, int(np.ceil(QUIET_FRACTION * len(ix)))))
        a = acc[ix]
        cs = np.vstack([np.zeros((1, 3)), np.cumsum(a, axis=0)])
        cs2 = np.vstack([np.zeros((1, 3)), np.cumsum(a * a, axis=0)])
        means = (cs[n:] - cs[:-n]) / n
        variances = np.maximum((cs2[n:] - cs2[:-n]) / n - means ** 2, 0)
        j = int(np.argmin(np.sqrt(variances).sum(axis=1)))
        v = a[j:j+n].mean(axis=0)
        roll = np.degrees(np.arctan2(v[1], v[2]))
        pitch = np.degrees(np.arctan2(-v[0], np.hypot(v[1], v[2])))
        tilt = np.degrees(np.arctan2(np.hypot(v[0], v[1]), v[2]))
        rows.append((float(np.mean(t[ix[j:j+n]])), roll, pitch, tilt))
    if len(rows) < 2:
        raise ValueError('Insufficient quiet segments for orientation plots')
    r = pd.DataFrame(rows, columns=['time_s', 'roll_deg', 'pitch_deg', 'tilt_deg'])
    r['roll_deg'] = np.degrees(np.unwrap(np.radians(r['roll_deg'])))
    for name in ('roll', 'pitch', 'tilt'):
        raw = r[f'{name}_deg']
        r[f'{name}_relative_deg'] = raw - raw.iloc[:min(5, len(r))].mean()
        r[f'{name}_smooth_deg'] = (r[f'{name}_relative_deg']
                                   .rolling(SMOOTH_SEGMENTS, center=True, min_periods=1).median()
                                   .rolling(SMOOTH_SEGMENTS, center=True, min_periods=1).mean())
    return r

def plot_orientation(r, smooth, outdir, x_mode):
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    x = r['time_s'] if x_mode == 'Time' else r['loading_cycles']
    for name, color in zip(('roll', 'pitch', 'tilt'), ('#c93333', '#268a50', '#2454bd')):
        column = f'{name}_{"smooth" if smooth else "relative"}_deg'
        ax.plot(x, r[column], color=color, linewidth=0.85, label=name.capitalize())
    ax.set_xlabel('Time (s)' if x_mode == 'Time' else 'Loading cycles, N (–)')
    ax.set_ylabel('Relative orientation angle (°)')
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    ax.tick_params(direction='in')
    export_figure(fig, f'Orientation_{"Smoothed" if smooth else "Raw"}_vs_{x_mode}', outdir)

def estimate_cutoff(p, fs):
    nyquist = fs / 2
    max_cutoff = MAX_CUTOFF_NYQUIST_FRACTION * nyquist
    if not 0 < SPECTRAL_ENERGY_FRACTION < 1:
        raise ValueError('SPECTRAL_ENERGY_FRACTION must be between zero and one')
    if max_cutoff <= MIN_CUTOFF_HZ:
        raise ValueError('Sampling frequency is too low for the configured cutoff limits')
    powers = []
    freq = None
    for channel in ACC:
        signal = p[f'{channel}_detrended'].to_numpy(dtype=float)
        scale = np.max(np.abs(signal))
        normalized = signal / scale if scale > 0 else np.zeros_like(signal)
        f, power = periodogram(normalized, fs=fs, window='hann',
                               detrend=False, scaling='spectrum')
        if freq is None:
            freq = f
        powers.append(power)
    mean_power = np.mean(powers, axis=0)
    valid = (freq > 0) & (freq <= max_cutoff)
    freq = freq[valid]
    mean_power = mean_power[valid]
    if not len(freq) or np.sum(mean_power) <= 0:
        raise ValueError('No measurable acceleration spectrum; specify a manual cutoff')
    cumulative = np.cumsum(mean_power) / np.sum(mean_power)
    index = min(int(np.searchsorted(cumulative, SPECTRAL_ENERGY_FRACTION)), len(freq)-1)
    estimated = float(np.clip(freq[index], MIN_CUTOFF_HZ, max_cutoff))
    return estimated, float(cumulative[index])

def process_file(path, outdir):
    df, t, fs = load_data(path)
    if LOADING_FREQUENCY_HZ is None or LOADING_FREQUENCY_HZ <= 0:
        raise ValueError('Set LOADING_FREQUENCY_HZ > 0 to produce loading-cycle plots.')
    if LOWPASS_CUTOFF_HZ is not None and not 0 < LOWPASS_CUTOFF_HZ < fs/2:
        raise ValueError(f'Low-pass cutoff must be below Nyquist ({fs/2:.3f} Hz)')
    outdir.mkdir(parents=True, exist_ok=True)
    figure_types = ('Raw_vs_Filtered_Acceleration', 'Raw_vs_Filtered_Angular_Velocity',
                    'Orientation_Raw', 'Orientation_Smoothed')
    for old_spectrum in ('FFT_Power_Spectra.png', 'FFT_Power_Spectra.tiff'):
        (outdir / old_spectrum).unlink(missing_ok=True)
    for stem in figure_types:
        for suffix in ('', '_vs_Time', '_vs_Loading_Cycles'):
            for ext in ('png', 'tiff'):
                (outdir / f'{stem}{suffix}.{ext}').unlink(missing_ok=True)
    p = pd.DataFrame({'time_s': t, 'loading_cycles': t * LOADING_FREQUENCY_HZ})
    report = []
    for c in CHANNELS:
        raw = df[c].to_numpy(dtype=float)
        despiked, flags, window = hampel_interpolate(raw, t, fs)
        detrended = quadratic_detrend(despiked, t)
        for name, values in [('raw', raw), ('despiked', despiked),
                             ('detrended', detrended)]:
            p[f'{c}_{name}'] = values
        p[f'{c}_outlier'] = flags
        report.append({'channel': c, 'outliers': int(flags.sum()),
                       'outlier_percent': 100*flags.mean(),
                       'hampel_window_samples': window})
    if LOWPASS_CUTOFF_HZ is None:
        cutoff, energy_fraction = estimate_cutoff(p, fs)
        method = 'automatic_normalized_acceleration_fft_energy'
    else:
        cutoff, energy_fraction = float(LOWPASS_CUTOFF_HZ), np.nan
        method = 'manual_override'
    sos = butter(FILTER_ORDER, cutoff, btype='lowpass', fs=fs, output='sos')
    for channel in CHANNELS:
        p[f'{channel}_filtered'] = sosfiltfilt(sos, p[f'{channel}_detrended'].to_numpy())
    pd.DataFrame([{'cutoff_hz': cutoff, 'method': method,
                   'target_spectral_energy_fraction': SPECTRAL_ENERGY_FRACTION,
                   'achieved_spectral_energy_fraction': energy_fraction,
                   'sampling_frequency_hz': fs}]).to_csv(
                       outdir / 'Filter_Cutoff_Report.csv', index=False)
    total_outliers = sum(item['outliers'] for item in report)
    total_samples = len(p) * len(CHANNELS)
    pd.DataFrame([{'total_outliers': total_outliers, 'total_samples': total_samples,
                   'combined_outlier_percent': 100 * total_outliers / total_samples}]
                 ).to_csv(outdir / 'Combined_Outlier_Summary.csv', index=False)
    p.to_csv(outdir/'SmartRock_cleaned_signals.csv', index=False)
    pd.DataFrame(report).to_csv(outdir/'Outlier_Report.csv', index=False)
    for x_mode in ('Time', 'Loading_Cycles'):
        plot_raw_filtered(t, p, ACC, ACCELERATION_UNIT,
                          'Raw_vs_Filtered_Acceleration', outdir, x_mode)
        plot_raw_filtered(t, p, GYRO, GYROSCOPE_UNIT,
                          'Raw_vs_Filtered_Angular_Velocity', outdir, x_mode)
    if QUIET_SEGMENT_ORIENTATION:
        r = quiet_orientation(t, p)
        r['loading_cycles'] = r.time_s * LOADING_FREQUENCY_HZ
        r.to_csv(outdir/'Quiet_Segment_Orientation.csv', index=False)
        for x_mode in ('Time', 'Loading_Cycles'):
            plot_orientation(r, False, outdir, x_mode)
            plot_orientation(r, True, outdir, x_mode)
    print(f'Processed {path} | fs={fs:.2f} Hz | f_loading={LOADING_FREQUENCY_HZ:g} Hz | cutoff={cutoff:.3f} Hz ({method})')
    print(f'Figures saved to: {outdir.resolve()}')

def main():
    process_file(INPUT_FILE, OUTPUT_DIR)

if __name__ == '__main__':
    main()
