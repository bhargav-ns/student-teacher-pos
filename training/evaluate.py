import torch
from models.StudentPOSTagger import StudentPOSTagger
from data.dataset import sentences, word_to_ix, ix_to_tag, tag_to_ix

student_model = StudentPOSTagger(len(word_to_ix), len(tag_to_ix))
student_model.eval()  # Put the student model in evaluation mode

def evaluate():
    with torch.no_grad():
        for sentence, _ in sentences:
            sentence_in = torch.tensor([word_to_ix[word] for word in sentence.split()], dtype=torch.long)
            output = student_model(sentence_in)
            predicted_tags_indices = torch.argmax(output, dim=1)
            predicted_tags = [ix_to_tag[index.item()] for index in predicted_tags_indices]

            print(f"Sentence: {sentence}")
            print(f"Predicted Tags: {predicted_tags}\n")

if __name__ == "__main__":
    evaluate()