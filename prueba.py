import os
import torch
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
import gc
import torch
import torchaudio
import torchaudio.transforms as T

# Configuración de la semilla para reproducibilidad
random_seed = 42
torch.manual_seed(random_seed)

# Parámetros de configuración inicial
samplerate = 22050
init_batch_size = 20
init_num_epochs = 10
init_lr = 0.0005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Función para extraer el género de cada archivo
def parse_genres(fname):
    parts = fname.split('/')[-1].split('.')[0]
    return parts

# Clase para cargar y manejar el dataset
class MusicDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.files = []
        for c in os.listdir(root):
            # Agrega cada archivo .wav a la lista con su path completo
            self.files += [os.path.join(root, c, fname) for fname in os.listdir(os.path.join(root, c)) if fname.endswith('.wav')]
        self.classes = list(set(parse_genres(fname) for fname in self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        genre = parse_genres(fpath)
        class_idx = self.classes.index(genre)
        audio = torchaudio.load(fpath)[0]
        return audio, class_idx

# Cargar el dataset
data_dir = './genres_5sec'  # Asegúrate de que esta ruta sea correcta
dataset = MusicDataset(data_dir)

# Crear los pesos de las clases para balancear el dataset
labels = [dataset[i][1] for i in range(len(dataset))]
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Dividir el dataset en entrenamiento, validación y prueba usando StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=random_seed)
for train_idx, temp_idx in split.split(range(len(dataset)), labels):
    train_dataset = Subset(dataset, train_idx)
    temp_dataset = Subset(dataset, temp_idx)

val_test_labels = [labels[i] for i in temp_idx]
split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_seed)
for val_idx, test_idx in split.split(temp_idx, val_test_labels):
    val_dataset = Subset(dataset, [temp_idx[i] for i in val_idx])
    test_dataset = Subset(dataset, [temp_idx[i] for i in test_idx])

# Creación de data loaders para cada conjunto
train_loader = DataLoader(train_dataset, batch_size=init_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=init_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=init_batch_size, shuffle=False)

# Define a model class to experiment with different architectures
class ExperimentNN(nn.Module):
    def __init__(self, input_size, num_classes, layer_size, hidden_layers):
        super(ExperimentNN, self).__init__()
        print(f"Initializing model with {hidden_layers} hidden layers and {layer_size} layer size")
        self.fc_layers = nn.ModuleList()

        # Crear las capas ocultas
        for layer in range(hidden_layers):
            if layer == 0:
                self.fc_layers.append(nn.Linear(input_size, layer_size))  # Capa inicial
            else:
                self.fc_layers.append(nn.Linear(layer_size, layer_size))  # Capas intermedias
        
        # Capa de salida
        self.fc_layers.append(nn.Linear(layer_size, num_classes))

    def forward(self, x):
        for fc in self.fc_layers[:-1]:  # Ignorar la última capa
            x = F.relu(fc(x))  # Activación ReLU
        x = self.fc_layers[-1](x)  # Capa de salida (sin activación)
        return x  # El modelo retorna logits de tamaño [batch_size, num_classes]

# Modificar la función de entrenamiento

def train_model(model, criterion, optimizer, epochs, train_loader, val_loader, device):
    print("Starting training...")
    best_loss = float("inf")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}...")

        # Training loop
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Asegurarse que la salida es de la forma [batch_size, num_classes]
            outputs = model(inputs)
            print(f"Outputs shape: {outputs.shape}")  # Asegúrate de que sea [20, 10]

            loss = criterion(outputs, labels)  # La forma de outputs debe ser [batch_size, num_classes]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Train loss: {train_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                print(f"Validation outputs shape: {outputs.shape}")  # Asegúrate de que sea [20, 10]
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Validation loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            print(f"New best validation loss: {best_loss:.4f}")

    return best_loss

# Evaluación final en el conjunto de test
print("Evaluating best model on test set...")

# Hiperparámetros
input_size = 128 * 87  # Tamaño de entrada dependiendo del tamaño del espectrograma
num_classes = 10  # Número de clases (géneros musicales)
layer_size = 64  # Tamaño de las capas ocultas
layers = 3  # Número de capas en el modelo
learning_rate = 0.0005
weight_decay = 1e-4

# Inicializar el modelo, el criterio y el optimizador
model = ExperimentNN(input_size, num_classes, layer_size, layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Entrenar el modelo
model = train_model(model, criterion, optimizer, init_num_epochs, train_loader, val_loader, device)
model.eval()
test_loss = 0.0
correct = 0
total = len(test_loader.dataset)

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        print(f"Test outputs shape: {outputs.shape}")  # Asegúrate de que sea [20, 10]

        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # Obtener las predicciones
        predicted = outputs.argmax(dim=1)  # Obtiene el índice de la clase predicha

        correct += (predicted == labels).sum().item()

test_loss /= len(test_loader)
accuracy = 100 * correct / total

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")