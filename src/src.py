# Creating Neural Network (MLP) using Sequential Model

## Layers = {ReLU, Sigmoid}
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.Sigmoid(),
    nn.Linear(128,10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
epochs = []
all_losses = []

for epoch in range(num_epochs):
    
    epochs.append(epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}]",end="")
    print("xxxxxxxxx",end="")
    for images, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    print("xxxxxxxxx",end="")
    # Print the loss for every epoch
    print(f", Loss: {loss.item():.4f}")
    all_losses.append(loss.item())
    
    
plt.plot(epochs,all_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


model.eval()
total_correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / len(test_data)
acc.append(accuracy)
layers.append("RS")
print(f"Test Accuracy: {accuracy*100:.2f}%")


## LAYERS = {ReLU, Tanh}

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.Tanh(),
    nn.Linear(128,10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
epochs = []
all_losses = []

for epoch in range(num_epochs):
    
    epochs.append(epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}]",end="")
    print("xxxxxxxxx",end="")
    for images, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    print("xxxxxxxxx",end="")
    # Print the loss for every epoch
    print(f", Loss: {loss.item():.4f}")
    all_losses.append(loss.item())
    
    
plt.plot(epochs,all_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

model.eval()
total_correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / len(test_data)
acc.append(accuracy)
layers.append("RT")
print(f"Test Accuracy: {accuracy*100:.2f}%")


## LAYERS = {Sigmoid, Tanh}

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.Sigmoid(),
    nn.Linear(256, 128),
    nn.Tanh(),
    nn.Linear(128,10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
epochs = []
all_losses = []

for epoch in range(num_epochs):
    
    epochs.append(epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}]",end="")
    print("xxxxxxxxx",end="")
    for images, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    print("xxxxxxxxx",end="")
    # Print the loss for every epoch
    print(f", Loss: {loss.item():.4f}")
    all_losses.append(loss.item())
    
    
plt.plot(epochs,all_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

model.eval()
total_correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total_correct += (predicted == labels).sum().item()

accuracy = total_correct / len(test_data)
acc.append(accuracy)
layers.append("ST")
print(f"Test Accuracy: {accuracy*100:.2f}%")