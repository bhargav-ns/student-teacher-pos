import torch.nn as nn
import torch.nn.functional as F

class StudentPOSTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=32, hidden_dim=64):
        super(StudentPOSTagger, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.embeddings(sentence)
        out = F.relu(self.fc(embeds))
        tag_space = self.output(out)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores