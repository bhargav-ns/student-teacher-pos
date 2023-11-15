# POS Tagger with Teacher-Student Network

This project implements a Part-of-Speech (POS) Tagger in PyTorch using a teacher-student network approach enhanced with First-Order Logic (FOL) rules. It's designed to demonstrate how a neural network (student) can learn POS tagging effectively by mimicking a more complex model (teacher) and incorporating structured linguistic knowledge through FOL rules.

## Project Structure

pos-tagger/
│
├── models/                  # Neural network models
├── data/                    # Data processing modules
├── rules/                   # FOL rules for POS tagging
├── training/                # Training and evaluation scripts
├── utils/                   # Utility functions
├── tests/                   # Unit tests
├── config.py                # Configuration settings
├── requirements.txt         # Project dependencies
└── main.py                  # Main script to run the POS tagger

## Getting Started
### Prerequisites
Python 3.x
PyTorch

### Installation
Clone the repository:
git clone https://github.com/your-repository/student-teacher-pos.git
cd pos-tagger

### Install the required packages:
pip install -r requirements.txt
Download the Universal Dependencies English Web Treebank (UD_English-EWT) dataset and place it in the data/UD_English-EWT directory. Ensure that the dataset files are named appropriately as expected in the data/dataset.py script.

### Usage
Run the main script to train and evaluate the POS tagger:
python main.py

### Components
Models: Teacher and student neural network models for POS tagging.
Data: Scripts for loading and preprocessing the POS tagging dataset.
Rules: Implementation of FOL rules applied to the teacher model's predictions.
Training: Scripts for training the teacher and student models, and for evaluating the student model's performance.
Utils: Utility functions for common operations like saving and loading models.
Tests: Unit tests for ensuring the reliability of the models.

### Configuration
Modify the config.py file to change the model parameters, training settings, and file paths as needed.
