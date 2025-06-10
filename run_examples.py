import torch
import torch.nn as nn
from sophia_optimizer import Sophia

# --- Setup Model and Data ---
class ClassifierNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, BATCH_SIZE = 20, 64, 5, 128
X_class = torch.randn(BATCH_SIZE, INPUT_DIM)
y_class = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))

# --- Initialize ---
model = ClassifierNN(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES)
optimizer = Sophia(model.parameters(), lr=1e-3, rho=0.03) # Defaults to hessian_computation_type='gnb'
criterion = nn.CrossEntropyLoss()

# --- Training Loop ---
print("Training classifier with Sophia-G...")
for epoch in range(1001):
    optimizer.zero_grad()
    logits = model(X_class)
    loss = criterion(logits, y_class)
    
    # Must retain graph for the GNB estimator's second backward pass
    loss.backward(retain_graph=True)
    
    optimizer.step(logits=logits)
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:04d} | Loss: {loss.item():.4f}")


# --- Setup Model and Data ---
class RegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

INPUT_DIM, HIDDEN_DIM, BATCH_SIZE = 20, 64, 128
X_reg = torch.randn(BATCH_SIZE, INPUT_DIM)
y_reg = torch.randn(BATCH_SIZE, 1) 

# --- Initialize ---
model = RegressionNN(INPUT_DIM, HIDDEN_DIM)
optimizer = Sophia(
    model.parameters(), 
    lr=1e-4, 
    rho=0.05,
    hessian_computation_type='hutchinson' # Set explicitly
)
criterion = nn.MSELoss()

# --- Training Loop ---
print("Training regressor with Sophia-H...")
for epoch in range(1001):
    optimizer.zero_grad()
    
    # The closure only returns the loss.
    def closure():
        outputs = model(X_reg)
        loss = criterion(outputs, y_reg)
        return loss

    current_loss = optimizer.step(closure=closure)
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:04d} | Loss: {current_loss.item():.4f}")