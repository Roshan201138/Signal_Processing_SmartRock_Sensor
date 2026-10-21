# SmartRock signal processing

Python scripts for processing SmartRock accelerometer and gyroscope measurements obtained during cyclic triaxial loading and compaction of unbound granular materials. The scripts create `Cyclic_loading_output/` and `Compaction_output/` automatically. 
```
## Repository structure
├── Compaction_Data/
├── Cyclic_loading_data/
├── Compaction.py
├── Cyclic_loading.py
├── README.md
└── requirements.txt

## Installation

Use Python 3.10 or newer. From the repository directory:

```bash
python -m pip install -r requirements.txt
```

## Input data

Both scripts require these seven columns (case-sensitive for compaction): timestamp,accX,accY,accZ,gyroX,gyroY,gyroZ

Timestamps are interpreted as **milliseconds** by default. Acceleration is assumed to be in **g**, and angular velocity in **degrees per second**. Change the constants near the top of each script if your data use different units. Text files should have a header row and a delimiter supported by `pandas.read_csv` (comma-separated files are recommended).

### Cyclic triaxial loading

Place the experimental file at:

```text
Cyclic_loading_data/Cyclic_loading.txt
```

Run:

```bash
python Cyclic_loading.py
```

The loading frequency is **2 Hz** by default (`LOADING_FREQUENCY_HZ`). Edit it to match your experiment. Outputs appear in `Cyclic_loading_output/`.

### Tamping and vibratory compaction

Place one data file for each of Layers 4, 5, and 6 in `Compaction_Data/`. File names must contain `Layer4`, `Layer5`, or `Layer6` (separators such as `Layer_4` are also accepted). Supported extensions are `.txt`, `.csv`, `.xlsx`, and `.xls`.

Run:

```bash
python Compaction.py
```

