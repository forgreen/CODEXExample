# CODEXExample

This repository contains a sample dataset (`data.csv`) and a Python script
(`analysis.py`) that performs the following tasks:

- Loads the CSV with pandas
- Cleans the data by removing missing values and converting columns to numeric
- Computes basic statistics (mean, standard deviation, max, min)
- Creates score distribution plots with Matplotlib
- Generates natural language summaries comparing each student's score to the
  class average
- Saves the plots and summaries into `report.pdf`

## Usage

Install the required dependencies (pandas and matplotlib) and then run:

```bash
python3 analysis.py
```

The script prints statistics and summary text to the console and creates
`report.pdf` in the same directory.
