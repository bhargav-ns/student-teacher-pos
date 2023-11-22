
import os
from training.train import train_teacher, train_student
from training.evaluate import evaluate
import utils.utils as utils
import config

def main():
    # Create directories for saving models if they don't exist
    utils.create_dir_if_not_exists(os.path.dirname(config.TEACHER_MODEL_PATH))
    utils.create_dir_if_not_exists(os.path.dirname(config.STUDENT_MODEL_PATH))

    # Train the teacher model
    print("Training the teacher model...")
    train_teacher()

    # Train the student model
    print("Training the student model...")
    train_student()

    # Evaluate the student model
    print("Evaluating the student model...")
    evaluate()

if __name__ == '__main__':
    main()