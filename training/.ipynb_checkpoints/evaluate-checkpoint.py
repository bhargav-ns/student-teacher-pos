import torch
from models.StudentPOSTagger import StudentPOSTagger
from data.dataset import dev_data, word_to_ix_dev, ix_to_tag_dev, tag_to_ix_dev
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

student_model = StudentPOSTagger(len(word_to_ix_dev), len(tag_to_ix_dev))
student_model.eval()  # Put the student model in evaluation mode

def evaluate():
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for sentence, tags in dev_data:
            sentence_in = torch.tensor([word_to_ix_dev[word] for word in sentence.split()], dtype=torch.long)
            targets = torch.tensor([tag_to_ix_dev[tag] for tag in tags], dtype=torch.long)
            output = student_model(sentence_in)
            
            
            predicted_tags_indices = torch.argmax(output, dim=1)
            predicted_tags = [ix_to_tag_dev[index.item()] for index in predicted_tags_indices]
            
            # Store predictions and targets for later computation of accuracy and F1 score
            all_predictions.extend(predicted_tags_indices.tolist())
            all_targets.extend(targets.tolist())
            
            
            
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    
    print(f"Sample Sentence: {sentence}")
    print(f"Sample Predicted Tags: {predicted_tags}\n")
    
if __name__ == "__main__":
    evaluate()