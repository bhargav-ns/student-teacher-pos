import torch
import torch.optim as optim
import torch.nn.functional as F
from models.TeacherPOSTagger import TeacherPOSTagger
from models.StudentPOSTagger import StudentPOSTagger
from data.dataset import sentences, word_to_ix, tag_to_ix, ix_to_tag
from rules.fol_rules import apply_logic_rules

# Initialize models
teacher_model = TeacherPOSTagger(len(word_to_ix), len(tag_to_ix))
student_model = StudentPOSTagger(len(word_to_ix), len(tag_to_ix))

# Loss function and optimizer
loss_function = torch.nn.NLLLoss()
teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=0.1)
student_optimizer = optim.SGD(student_model.parameters(), lr=0.1)

# Training loop for the teacher model
def train_teacher():
    teacher_model.train()
    for epoch in range(50):
        for sentence, tags in sentences:
            sentence_in = torch.tensor([word_to_ix[word] for word in sentence.split()], dtype=torch.long)
            targets = torch.tensor([tag_to_ix[tag] for tag in tags], dtype=torch.long)

            teacher_model.zero_grad()
            tag_scores = teacher_model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            teacher_optimizer.step()

# Training loop for the student model
def train_student():
    teacher_model.eval()
    student_model.train()
    for epoch in range(50):
        for sentence, _ in sentences:
            sentence_in = torch.tensor([word_to_ix[word] for word in sentence.split()], dtype=torch.long)

            # Get teacher predictions and apply logic rules
            with torch.no_grad():
                teacher_output_raw = teacher_model(sentence_in)
                teacher_output = apply_logic_rules(teacher_output_raw, sentence.split(), tag_to_ix, ix_to_tag)

            # Get student predictions
            student_output = student_model(sentence_in)

            # Compute loss and update student model
            loss = F.kl_div(student_output, teacher_output, reduction='batchmean')
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()

if __name__ == "__main__":
    train_teacher()
    train_student()