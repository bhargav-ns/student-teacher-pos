import unittest
import torch
from models.TeacherPOSTagger import TeacherPOSTagger
from models.StudentPOSTagger import StudentPOSTagger

class TestPOSTaggerModels(unittest.TestCase):

    def test_teacher_model_output(self):
        vocab_size = 10  # Example vocabulary size
        tagset_size = 5  # Example tagset size
        model = TeacherPOSTagger(vocab_size, tagset_size)
        input = torch.randint(0, vocab_size, (1, 5))  # Example input
        output = model(input)
        self.assertEqual(output.shape, (5, tagset_size))

    def test_student_model_output(self):
        vocab_size = 10
        tagset_size = 5
        model = StudentPOSTagger(vocab_size, tagset_size)
        input = torch.randint(0, vocab_size, (1, 5))
        output = model(input)
        self.assertEqual(output.shape, (5, tagset_size))

if __name__ == '__main__':
    unittest.main()