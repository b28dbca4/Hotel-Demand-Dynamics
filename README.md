# AI Impact on Society Analysis

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.1.1-blue.svg)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.1-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-4.0.7-orange.svg)](https://jupyter.org/)

## Overview

This project analyzes the impact of artificial intelligence on society using data from the Kaggle dataset [The Impact of Artificial Intelligence on Society](https://www.kaggle.com/datasets/ardayavuzkeskin/the-impact-of-artificial-intelligence-on-society). It encompasses data collection, exploration, preprocessing, exploratory data analysis, feature engineering, model training, evaluation, and final analysis and conclusions.

## Features

- **Data Collection and Exploration**: Automated scripts for gathering and exploring AI impact data.
- **Data Preprocessing**: Cleaning and transforming raw data for analysis.
- **Exploratory Data Analysis (EDA)**: Visualizing and understanding data patterns related to AI's societal impact.
- **Feature Engineering**: Creating relevant features for predictive modeling.
- **Model Training**: Implementing machine learning models to analyze AI impacts.
- **Model Evaluation**: Assessing model performance with various metrics.
- **Final Analysis and Conclusions**: Deriving insights and conclusions from the analysis.

## Installation

### Prerequisites

- Python 3.11 or higher
- Conda (recommended for environment management)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/b28dbca4/AI-Impact-on-Society-Analysis.git
   cd AI-Impact-on-Society-Analysis
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate ds-env
   ```

   Alternatively, install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The project is structured around Jupyter notebooks for step-by-step execution:

1. **01_data_collection_exploration.ipynb**: Collect and explore raw data.
2. **02_data_preprocessing.ipynb**: Preprocess the data.
3. **03_eda_and_question.ipynb**: Perform exploratory data analysis.
4. **04_feature_engineering.ipynb**: Engineer features.
5. **05_model_training.ipynb**: Train predictive models.
6. **06_model_evaluation.ipynb**: Evaluate model performance.
7. **07_final_analysis_conclusions.ipynb**: Generate final analysis and conclusions.

To run the notebooks, start Jupyter:
```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and execute the notebooks in order.

## Project Structure

```
.
├── data/
│   ├── final/        # Final processed data
│   ├── processed/    # Intermediate processed data
│   └── raw/          # Raw data
├── notebooks/        # Jupyter notebooks for analysis
│   ├── 01_data_collection_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_eda_and_question.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_final_analysis_conclusions.ipynb
├── reports/
│   ├── figures/      # Generated figures
│   │   ├── eda_plots/
│   │   └── model_results/
│   └── presentations/ # Presentation slides
├── src/              # Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_cleaner.py      # Data cleaning utilities
│   │   ├── data_loader.py       # Data loading functions
│   │   └── feature_engineering.py # Feature engineering tools
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_evaluator.py   # Model evaluation scripts
│   │   ├── model_interpretation.py # Model interpretation tools
│   │   └── model_trainer.py     # Model training functions
│   ├── test/
│   │   ├── __init__.py
│   │   ├── test_data_processing.py # Tests for data processing
│   │   └── test_models.py       # Tests for models
│   ├── utils/
│   │   └── config.py            # Configuration settings
│   └── visualization/
│       ├── __init__.py
│       ├── dashboard.py         # Dashboard creation
│       └── plot_utils.py        # Plotting utilities
├── environment.yml   # Conda environment file
├── requirements.txt  # Python dependencies
├── LICENSE           # License file
└── README.md
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature.
3. Make your changes and add tests if applicable.
4. Submit a pull request.

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Contact

For questions or issues, please open an issue on GitHub. 
