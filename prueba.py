import os
import gc
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as tt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Configuración de la semilla para reproducibilidad
random_seed = 42
torch.manual_seed(random_seed)

# Parámetros de configuración inicial
samplerate = 22050
init_batch_size = 20        
init_num_epochs = 30  # Incremento de épocas
init_lr = 0.0005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Función para extraer el género de cada archivo
def parse_genres(fname):
    parts = fname.split('/')[-1].split('.')[0]
    return parts

# Clase para cargar y manejar el dataset
class MusicDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.files = []
        self.transform = transform
        for c in os.listdir(root):
            self.files += [os.path.join(root, c, fname) for fname in os.listdir(os.path.join(root, c)) if fname.endswith('.wav')]
        self.classes = list(set(parse_genres(fname) for fname in self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = self.files[idx]
        genre = parse_genres(fpath)
        class_idx = self.classes.index(genre)
        audio, sr = torchaudio.load(fpath)
        if self.transform:
            audio = self.transform(audio)
        return audio, class_idx

# Transformación para convertir audio a Mel-spectrograma
spectrogram_transform = tt.MelSpectrogram(sample_rate=samplerate, n_fft=1024, hop_length=512)

# Cargar el dataset con la transformación
data_dir = './genres_5sec'
dataset = MusicDataset(data_dir, transform=spectrogram_transform)

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

# Define a model class with ReLU activation and Dropout layers
class ExperimentNN(nn.Module):
    def __init__(self, input_size, num_classes, layer_size, layers):
        super(ExperimentNN, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% probability

        for layer in range(layers - 1):
            if layer == 0:
                self.fc_layers.append(nn.Linear(input_size, layer_size))
            else:
                self.fc_layers.append(nn.Linear(layer_size, layer_size))
        
        # Output layer
        self.fc_layers.append(nn.Linear(layer_size, num_classes))

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Aplanar el espectrograma
        x = (x - x.mean()) / x.std()  # Normalización
        for fc in self.fc_layers[:-1]:  # Skip last layer
            x = self.dropout(F.relu(fc(x)))  # ReLU + Dropout
        x = self.fc_layers[-1](x)  # Output layer (no activation)
        return x

def train_model(model, criterion, optimizer, scheduler, epochs, train_loader, val_loader, device):
    best_loss = float("inf")
    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Validation loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            print(f"New best validation loss: {best_loss:.4f}")

        # Update learning rate with scheduler
        scheduler.step(val_loss)
        
        # Clear cache and collect garbage
        gc.collect()
        torch.cuda.empty_cache()

    return best_loss

# Definir los parámetros del modelo basados en el tamaño del espectrograma (frequencies x time_steps)
frequencies = 128  # Mel-spectrogram reduce dimensionalidad
time_steps = (samplerate * 5) // 512 + 1
input_size = frequencies * time_steps  # Tamaño de entrada basado en el espectrograma aplanado

num_classes = len(dataset.classes)  # Número de clases (géneros musicales)

# Define nuevos valores de hiperparámetros para experimentar
layers_list = [2, 3, 5, 7]       # Pruebas con más capas
sizes_list = [32, 64, 128, 256]  # Pruebas con más unidades en cada capa

best_val_loss = float("inf")
best_val_accuracy = 0
learning_rate = 0.0005
weight_decay = 1e-4
best_model = None

# Loop de experimentación
for layers in layers_list:
    for size in sizes_list:
        print(f"Experimentando con {layers} capas y {size} unidades por capa...")
        
        # Inicializar el modelo con la configuración actual
        model = ExperimentNN(input_size=input_size,
                             num_classes=num_classes,
                             layer_size=size,
                             layers=layers).to(device)

        # Definir el criterio y optimizador con los pesos de las clases
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Define el scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # Llama a train_model pasándole el scheduler
        validation_loss = train_model(
            model, criterion, optimizer, scheduler=scheduler, epochs=30,
            train_loader=train_loader, val_loader=val_loader, device=device
        )

        # Calcular precisión de validación
        model.eval()
        correct = 0
        total = len(val_loader.dataset)
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
        val_accuracy = 100 * correct / total

        print(f"Loss de validación para {layers} capas y {size} unidades: {validation_loss}")
        print(f"Precisión de validación: {val_accuracy:.2f}%")

        # Guardar el modelo con mejor precisión y menor pérdida
        if val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and validation_loss < best_val_loss):
            best_val_loss = validation_loss
            best_val_accuracy = val_accuracy
            best_model = model
            print(f"Nuevo mejor modelo encontrado con {layers} capas y {size} unidades.")

print(f"Mejor modelo: {layers} capas y {size} unidades, con precisión de validación de {best_val_accuracy:.2f}% y pérdida de validación de {best_val_loss:.4f}")

# Evaluación final en el conjunto de prueba
print("Evaluando el mejor modelo en el conjunto de prueba...")
best_model.eval()
test_loss = 0.0
correct = 0
total = len(test_loader.dataset)
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = best_model(inputs)

        loss = criterion(outputs, labels)
        test_loss += loss.item()

        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()

test_loss /= len(test_loader)
test_accuracy = 100 * correct / total

print(f"Loss en el conjunto de prueba: {test_loss:.4f}")
print(f"Precisión en el conjunto de prueba: {test_accuracy:.2f}%")