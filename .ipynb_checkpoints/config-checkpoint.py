# Path to the dataset
DATA_PATH = 'data/UD_English-EWT'

# Model parameters
TEACHER_MODEL_PARAMS = {
    'vocab_size': 10000,
    'tagset_size': 20,    
    'embedding_dim': 64,
    'hidden_dim': 128
}

STUDENT_MODEL_PARAMS = {
    'vocab_size': 10000, 
    'tagset_size': 20,    
    'embedding_dim': 32,
    'hidden_dim': 64
}

# Training parameters
NUM_EPOCHS = 30
LEARNING_RATE = 0.1

# File paths for saving and loading models
TEACHER_MODEL_PATH = './models/teacher_model.pth'
STUDENT_MODEL_PATH = './models/student_model.pth'