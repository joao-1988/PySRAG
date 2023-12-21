# PySRAG

This Python package provides tools for analyzing and processing data related to Severe Acute Respiratory Syndrome (SARS) and other respiratory viruses. It includes functions for data preprocessing, feature engineering, and training Gradient Boosting Models (GBMs) for binary or multiclass classification.

## Getting Started

These instructions will help you get started with using the PySRAG package.

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.10.12 installed
- Required Python packages (you can install them using `pip`):
  - `pandas==1.5.3`
  - `numpy==1.23.5`
  - `joblib==1.3.2`
  - `scikit-learn==1.2.2`
  - `lightgbm==4.0.0`

<!---
### Installation

You can install the PySRAG package using `pip`:

```bash
pip install PySRAG
```
--->

### Usage

Here's an example of how to use the SRAG package:

```python
from PySRAG import PySRAG

filepath = 'INFLUD23-07-08-2023.csv' # from https://opendatasus.saude.gov.br/dataset/srag-2021-a-2023

# Initialize the SRAG class
srag = PySRAG.SRAG(filepath)

# Generate training data
X, y = srag.generate_training_data(lag=None, objective='multiclass')

# Train a Gradient Boosting Model
trainer = PySRAG.GBMTrainer(objective='multiclass', eval_metric='multi_logloss')
trainer.fit(X, y)
```

<!---
For more detailed information and examples, please refer to the package documentation.

## Documentation

You can find the full documentation for the SRAG package in the [docs](docs/) directory.

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add new feature"`
4. Push to your branch: `git push origin feature/your-feature-name`
5. Create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the contributors and maintainers of the SRAG Analysis package.

Happy coding!
-->
