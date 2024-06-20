import torch
from torch import nn

class FrameworkClassifier(nn.Module):
    def __init__(self, num_frameworks, framework_embeddings):
        super(FrameworkClassifier, self).__init__()
        self.num_frameworks = num_frameworks
        self.framework_embeddings = framework_embeddings
        self.fc = nn.Linear(512, num_frameworks)

    def forward(self, x):
        return self.fc(x)
    
    def predict(self, x):
        with torch.no_grad():
            return self.framework_embeddings[torch.argmax(self.forward(x))]
    
    def train(self, x, y, epochs=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        return self