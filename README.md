# PySRAG

This Python package provides tools for analyzing and processing data related to Severe Acute Respiratory Syndrome (SARS) and other respiratory viruses. It includes functions for data preprocessing, feature engineering, and training Gradient Boosting Models (GBMs) for binary or multiclass classification.

## Getting Started

These instructions will help you get started with using the PySRAG package.

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3 installed
- Required Python packages (you can install them using `pip`):
  - `pandas==1.5.3`
  - `numpy==1.23.5`
  - `scikit-learn==1.2.2`
  - `lightgbm==4.0.0`

### Installation

You can install the PySRAG package using `pip`:

```python
pip install PySRAG
```

### Usage

Here's an example of how to use the SRAG package:

```python
from pysrag.data import SRAG
from pysrag.model import GBMTrainer

# from https://opendatasus.saude.gov.br/dataset/srag-2021-a-2024
filepath = 'https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2023/INFLUD23-16-10-2023.csv' 

# Initialize the SRAG class
srag = SRAG(filepath)

# Generate training data
inputs = ['REGIAO_LATITUDE', 'REGIAO_LONGITUDE', 'UF_LATITUDE'
        , 'UF_LONGITUDE', 'LATITUDE', 'LONGITUDE', 'POPULACAO', 'IDADE_ANO'
        , 'ANO_SEM_SIN_PRI']
target = ['POS_SARS2', 'POS_FLUA', 'POS_FLUB', 'POS_VSR']
residual_viruses = ['POS_PARA1', 'POS_PARA2', 'POS_PARA3', 'POS_PARA4',
                    'POS_ADENO', 'POS_METAP', 'POS_BOCA', 'POS_RINO', 'POS_OUTROS']

X, y = srag.generate_training_data(objective='multiclass', cols_X=inputs, col_y=target, residual_viruses=residual_viruses)

# Train a Gradient Boosting Model
trainer = GBMTrainer(objective='multiclass', eval_metric='multi_logloss')
trainer.fit(X, y)

# Get Prevalences
trainer.model.predict_proba(X)
array([[9.73523796e-05, 8.91182790e-04, 1.21236644e-01, 8.64260161e-01, 1.35146598e-02],
       [4.71281550e-03, 2.36337464e-05, 9.59325690e-01, 2.72306200e-02, 8.70724046e-03],
       [6.95816743e-04, 3.35154571e-05, 2.81288034e-04, 9.98876481e-01, 1.12898420e-04],
       ...,
       [4.62475587e-03, 2.82325172e-03, 3.81832162e-03, 1.39748287e-01, 8.48985384e-01],
       [4.62475587e-03, 2.82325172e-03, 3.81832162e-03, 1.39748287e-01, 8.48985384e-01],
       [1.13695780e-02, 1.17825387e-03, 1.04659501e-02, 9.74318052e-01, 2.66816576e-03]])
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

## Web Application

The PySRAG package includes a web application that allows users to interactively explore data related to Severe Acute Respiratory Syndrome (SARS) in Brazil. This web-based interface provides a practical way for users to visualize data without needing deep technical knowledge of Python or the underlying code.

### Accessing the Web Application

To access the web application, visit:

[PySRAG Web App](https://joaoflavio.shinyapps.io/Virus_Monitor/)

This link will take you to a hosted version of our application, equipped with preloaded data and features for easy exploration.
 
[![](webapp_PySRAG.png)](https://joaoflavio.shinyapps.io/Virus_Monitor/)

### Features

The web application offers the following features:

- **Data Visualization**: Interactive graphs display processed data, giving insights into the distribution of respiratory viruses.
- **Data Filtering**: Users can apply filters based on city and patient age to narrow down the data and focus on specific demographics or regions.

### How to Use

1. **Navigate to the Dashboard**: Start on the dashboard, which provides an overview of the visualizations.
2. **Apply Filters**: Use the filtering options to select specific cities or age ranges to view customized data visualizations.
4. **Explore Visualizations**: Interact with the visual data representations to gain deeper insights into the trends and patterns.

### Support

If you encounter any issues while using the web application or have suggestions for improvements, please submit an issue on our [GitHub page](https://github.com/joao-1988/PySRAG/issues).

This web application is designed to make the data analysis capabilities of the PySRAG package accessible to both technical and non-technical users, enhancing understanding and facilitating research on respiratory viruses.
