# Doping Detection AI Framework

## Table of Contents
1. [Introduction](#introduction)
2. [AI Framework Overview](#ai-framework-overview)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Model Architecture](#model-architecture)
6. [Training Process](#training-process)
7. [Evaluation](#evaluation)
8. [Usage](#usage)
9. [Web Application (Coming Soon)](#web-application-coming-soon)
10. [API Integration (Coming Soon)](#api-integration-coming-soon)
11. [Contributing](#contributing)
12. [License](#license)

## Introduction

The Doping Detection AI Framework is an open-source project aimed at developing advanced machine learning models for detecting potential doping in athletes. By leveraging medical parameters such as blood tests and biological passport data, this framework provides a robust foundation for anti-doping efforts in sports.

## AI Framework Overview

Our framework utilizes a hybrid deep learning and machine learning approach, combining:

- A RandomForestClassifier Algorithm imeplementing binary classification for the model to output how likely it is that the athlete has undergone doping.
- Feedforward Neural Networks for handling static athlete features
- Advanced regularization techniques to prevent overfitting
- Custom loss functions to address class imbalance

Key Features:
- Flexible data input pipeline
- Modular model architecture
- Comprehensive evaluation metrics
- Easy-to-use training and inference scripts

## Installation

```bash
git clone https://github.com/your-username/doping-detection-ai.git
cd doping-detection-ai
pip install -r requirements.txt
```

## Data Preparation

The framework expects two main types of data:

1. Static Data: Unchanging or slowly changing athlete features
2. Temporal Data: Time-series data from biological passports and regular tests

Example data format:

```python
# Static data
static_data = pd.DataFrame({
    'athlete_id': [1, 2, 3, ...],
    'age': [25, 30, 28, ...],
    'gender': ['M', 'F', 'M', ...],
    'sport': ['cycling', 'athletics', 'swimming', ...]
})

# Temporal data
temporal_data = pd.DataFrame({
    'athlete_id': [1, 1, 1, 2, 2, 2, ...],
    'test_date': ['2023-01-01', '2023-02-01', '2023-03-01', ...],
    'hemoglobin': [15.1, 15.3, 14.9, ...],
    'reticulocyte_percent': [1.2, 1.1, 1.3, ...]
})
```

For more details on data preparation, see `data_preparation.py`.

## Model Architecture

The core of our framework is the `DopingDetectionModel` class:

```python
class DopingDetectionModel(nn.Module):
    def __init__(self, static_dim, temporal_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(temporal_dim, hidden_dim, batch_first=True)
        self.fc_static = nn.Linear(static_dim, hidden_dim)
        self.fc_combined = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, static, temporal):
        lstm_out, _ = self.lstm(temporal)
        lstm_out = lstm_out[:, -1, :]
        static_out = self.fc_static(static)
        combined = torch.cat((lstm_out, static_out), dim=1)
        combined = self.dropout(combined)
        return torch.sigmoid(self.fc_combined(combined))
```

This architecture processes static and temporal data separately before combining them for the final prediction.

## Training Process

Our training process utilizes:
- K-fold cross-validation
- Adam optimizer with learning rate scheduling
- Binary Cross-Entropy loss
- Early stopping to prevent overfitting

To train the model:

```bash
python train_model.py --data_path /path/to/your/data --epochs 100 --lr 0.001
```

For advanced training options, refer to `train_model.py`.

## Evaluation

We provide a comprehensive evaluation suite in `evaluate_model.py`, including:

- ROC-AUC Score
- Precision-Recall Curve
- F1 Score
- Confusion Matrix

To evaluate a trained model:

```bash
python evaluate_model.py --model_path /path/to/saved/model --test_data /path/to/test/data
```

## Usage

After training, you can use the model for inference:

```python
from doping_detection import DopingDetectionModel, preprocess_data

# Load your trained model
model = DopingDetectionModel.load_from_checkpoint('path/to/model.ckpt')

# Preprocess your data
static_data, temporal_data = preprocess_data(your_raw_data)

# Make predictions
predictions = model(static_data, temporal_data)
```

## Web Application (Coming Soon)

We are developing a web interface for easy interaction with the AI model. This section will be updated with details on:

- User Interface
- Visualization of Results
- Data Upload Functionality

Stay tuned for updates!

## API Integration (Coming Soon)

We plan to provide a RESTful API for seamless integration with existing systems. This section will cover:

- API Endpoints
- Request/Response Formats
- Authentication
- Rate Limiting

Check back for updates on API availability and documentation.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details on how to get involved.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

