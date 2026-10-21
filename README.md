# SmartRock signal processing

Python scripts for processing SmartRock sensor accelerometer and gyroscope measurements obtained during cyclic triaxial loading and compaction of unbound granular materials. The scripts create `Cyclic_loading_output/` and `Compaction_output/` automatically. 
```
## Repository structure

```text
├── Compaction_Data/
├── Cyclic_loading_data/
├── Compaction.py
├── Cyclic_loading.py
├── README.md
└── requirements.txt
```

## Installation

Use Python 3.10 or newer. From the repository directory:

```bash
python -m pip install -r requirements.txt
```

## Usage

### Compaction analysis

```bash
python Compaction.py
```

### Cyclic loading analysis

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

