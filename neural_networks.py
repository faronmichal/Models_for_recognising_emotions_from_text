import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report

# data
df = pd.read_pickle("C:/Users/michal1/Desktop/magisterka/merged_training.pkl")

# remove duplicates
duplicates = df['text'].duplicated(keep=False).sum()
print(f"Znaleziono {duplicates} wystąpień powtarzających się wartości w kolumnie 'text'.")
df = df[~df['text'].duplicated(keep=False)]
print(f"Liczba wierszy po usunięciu wszystkich duplikatów: {len(df)}")

# plot emotions distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='emotions')

# undersampling
counts = df['emotions'].value_counts()
min_count = counts.min()
df = df.groupby('emotions').apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)

# plot after undersampling
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='emotions')

df.info()

# tokenization with tensorflow
text = df['text']
label = df['emotions']

encoder = LabelEncoder()
label_encoded = encoder.fit_transform(label)

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
x = pad_sequences(sequences, maxlen=50)
y = to_categorical(label_encoded)

# traintest split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=label_encoded, random_state=42)

# convert to PyTorch tensors
x_train_tensor = torch.LongTensor(x_train)
x_test_tensor = torch.LongTensor(x_test)
y_train_tensor = torch.FloatTensor(y_train)
y_test_tensor = torch.FloatTensor(y_test)




# DataLoader
batch_size = 64
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

CLASS_NAMES = None  

def plot_confusion_and_report(y_true, y_pred, class_names=None, normalize=True, title="Confusion matrix"):
    labels = np.arange(len(class_names)) if class_names is not None else None
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"{title} ({'normalized' if normalize else 'counts'})")
    tick_labels = class_names if class_names is not None else np.arange(cm.shape[0])
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # print values on the cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, val, ha="center", va="center")

    fig.tight_layout()
    plt.show()

    # Detailed per class precision/recall/F1
    print("\nClassification report:\n")
    try:
        print(classification_report(y_true, y_pred, target_names=class_names if class_names is not None else None, digits=4))
    except ValueError:
        print(classification_report(y_true, y_pred, digits=4))


# Feedforward model
class FeedForwardNN(nn.Module):
    def __init__(self, max_words, embedding_dim, max_len, num_classes):
        super(FeedForwardNN, self).__init__()
        self.embedding = nn.Embedding(max_words, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * max_len, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)           
        x = x.view(x.size(0), -1)       
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_words = 10000
embedding_dim = 100
max_len = 50
num_classes = y.shape[1]

model = FeedForwardNN(max_words, embedding_dim, max_len, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, torch.max(y_batch, 1)[1])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(y_batch, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f} - Accuracy: {acc:.2f}% - F1 Score: {f1:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(y_batch, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100 * correct / total
test_f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"\n✅ Test Accuracy: {test_acc:.2f}% - F1 Score: {test_f1:.4f}")
plot_confusion_and_report(all_labels, all_preds, class_names=CLASS_NAMES, normalize=True)


# CNN Model
class SingleBranchCNN(nn.Module):
    def __init__(self, max_words, embedding_dim, max_len, num_classes=4):
        super(SingleBranchCNN, self).__init__()
        self.embedding = nn.Embedding(max_words, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64, 64)
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)          
        x = x.permute(0, 2, 1)         
        x = self.conv(x)              
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = F.max_pool1d(x, kernel_size=x.size(2))  
        x = x.squeeze(2)               
        x = self.fc1(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_words = 10000
embedding_dim = 100
max_len = 50
num_classes = y.shape[1]

model = SingleBranchCNN(max_words, embedding_dim, max_len, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, torch.max(y_batch, 1)[1])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(y_batch, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f} - Accuracy: {acc:.2f}% - F1 Score: {f1:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(y_batch, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100 * correct / total
test_f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"\n✅ Test Accuracy: {test_acc:.2f}% - F1 Score: {test_f1:.4f}")
plot_confusion_and_report(all_labels, all_preds, class_names=CLASS_NAMES, normalize=True)






# CNN model with more hidden layers
class SingleBranchCNN(nn.Module):
    def __init__(self, max_words, embedding_dim, max_len, num_classes=4):
        super(SingleBranchCNN, self).__init__()
        self.embedding = nn.Embedding(max_words, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.embedding(x)                  
        x = x.permute(0, 2, 1)                 
        x = self.conv(x)                       
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = F.max_pool1d(x, kernel_size=x.size(2))  
        x = x.squeeze(2)                       
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_words = 10000
embedding_dim = 100
max_len = 50
num_classes = y.shape[1]

model = SingleBranchCNN(max_words, embedding_dim, max_len, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 25
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, torch.max(y_batch, 1)[1])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(y_batch, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f} - Accuracy: {acc:.2f}% - F1 Score: {f1:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(y_batch, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100 * correct / total
test_f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"\n✅ Test Accuracy: {test_acc:.2f}% - F1 Score: {test_f1:.4f}")
plot_confusion_and_report(all_labels, all_preds, class_names=CLASS_NAMES, normalize=True)





# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)                        
        out, _ = self.lstm(x)                         
        out = out[:, -1, :]                           
        out = self.dropout(out)
        out = F.relu(self.fc1(out))                   
        out = self.output(out)                        
        return out

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_words = 10000
embedding_dim = 100
hidden_dim = 128
max_len = 50
num_classes = y.shape[1]

model = LSTMModel(max_words, embedding_dim, hidden_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, torch.max(y_batch, 1)[1])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(y_batch, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f} - Accuracy: {acc:.2f}% - F1 Score: {f1:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(y_batch, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100 * correct / total
test_f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"\n✅ Test Accuracy: {test_acc:.2f}% - F1 Score: {test_f1:.4f}")
plot_confusion_and_report(all_labels, all_preds, class_names=CLASS_NAMES, normalize=True)









# LSTM Model with multiple layers and extra dense layers
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)                      
        out, _ = self.lstm(x)                      
        out = out[:, -1, :]                        
        out = self.dropout(out)
        out = F.relu(self.fc1(out))                
        out = self.dropout(out)
        out = F.relu(self.fc2(out))                
        out = self.output(out)                      
        return out

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_words = 10000
embedding_dim = 100
hidden_dim = 128
max_len = 50
num_classes = y.shape[1]

model = LSTMModel(max_words, embedding_dim, hidden_dim, num_classes, num_layers=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 25
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, torch.max(y_batch, 1)[1])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(y_batch, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f} - Accuracy: {acc:.2f}% - F1 Score: {f1:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(y_batch, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = 100 * correct / total
test_f1 = f1_score(all_labels, all_preds, average='weighted')
print(f"Test Accuracy: {test_acc:.2f}% - F1 Score: {test_f1:.4f}")
plot_confusion_and_report(all_labels, all_preds, class_names=CLASS_NAMES, normalize=True)



