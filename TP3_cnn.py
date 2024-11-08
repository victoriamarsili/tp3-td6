#!/usr/bin/env python
# coding: utf-8

# Universidad Torcuato Di Tella
# 
# Licenciatura en Tecnología Digital\
# **Tecnología Digital VI: Inteligencia Artificial**
# 

# In[1]:


import os
import torch
import torchaudio
import tarfile
import wandb
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio.datasets import GTZAN
from torch.utils.data import DataLoader
import torchaudio.transforms as tt
from torch.utils.data import random_split
import matplotlib
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 
# # TP3: Encodeador de música
# 
# 
# 
# ## Orden de pasos
# 
# 0. Elijan GPU para que corra mas rapido (RAM --> change runtime type --> T4 GPU)
# 1. Descargamos el dataset y lo descomprimimos en alguna carpeta en nuestro drive.
# 2. Conectamos la notebook a gdrive y seteamos data_dir con el path a los archivos.
# 3. Visualización de los archivos
# 4. Clasificación
# 5. Evaluación
# 
# 
# 

# In[ ]:


project_name='TP3-TD6'
username = "sansonmariano-universidad-torcuato-di-tella"
wandb.login(key="d2875c91a36209496ee81454cccd95ebe3dc948d")
wandb.init(project = project_name, entity = username)


# In[3]:


random_seed = 42

torch.manual_seed(random_seed)

# Definir parámetros
samplerate = 22050
data_dir = './genres_5sec'

init_batch_size = 20
init_num_epochs = 10
init_lr = 0.0005


# In[4]:


# Función para parsear géneros
def parse_genres(fname):
    parts = fname.split('/')[-1].split('.')[0]
    return parts

# Definir la clase del dataset
class MusicDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.files = []
        for c in os.listdir(root):
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


# In[5]:


from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset

# Cargar el dataset
dataset = MusicDataset(data_dir)

# Ensure labels match the number of samples
# Adjust if 'targets' is not the correct attribute
labels = dataset.classes if hasattr(dataset, 'targets') else [dataset[i][1] for i in range(len(dataset))]

# Initialize StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

# First, split into training (70%) and temp (30% for val + test)
for train_idx, temp_idx in split.split(range(len(dataset)), labels):
    train_dataset = Subset(dataset, train_idx)
    temp_dataset = Subset(dataset, temp_idx)

# Next, split the temp dataset (30%) into validation (15%) and test (15%)
val_test_labels = [labels[i] for i in temp_idx]
split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_idx, test_idx in split.split(temp_idx, val_test_labels):
    val_dataset = Subset(dataset, [temp_idx[i] for i in val_idx])
    test_dataset = Subset(dataset, [temp_idx[i] for i in test_idx])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=init_batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=init_batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=init_batch_size, shuffle=True, num_workers=0)


# In[6]:


list_files=os.listdir(data_dir)

classes = []

for file in list_files:

  name='{}/{}'.format(data_dir,file)

  if os.path.isdir(name):

    classes.append(file)


# ### 3. Visualización de los archivos

# In[7]:


def audio_to_spectrogram(waveform):
    # Ensure the waveform is in the correct shape
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)
    
    # Convert the waveform to a spectrogram
    spectrogram = tt.Spectrogram()(waveform)
    return spectrogram

def process_dataloader_to_spectrograms(dataloader):
    spectrograms = []
    
    for batch in dataloader:
        # Assuming the batch is a tuple (waveforms, labels) and waveforms are the audio data
        waveforms, labels = batch
        
        # Process each waveform in the batch
        batch_spectrograms = [audio_to_spectrogram(waveform) for waveform in waveforms]
        
        # Append to the list of spectrograms
        spectrograms.append((torch.stack(batch_spectrograms), labels))
    
    return spectrograms


# In[8]:


input_channels = 1  # for RGB images, or 1 for grayscale
num_classes = 10    # depends on your specific classification task

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[9]:


train_spectogram = process_dataloader_to_spectrograms(train_loader)
val_spectogram = process_dataloader_to_spectrograms(val_loader)
test_spectogram = process_dataloader_to_spectrograms(test_loader)


# ### Ejercicio 1

# Modelo que recibe el tamaño de la capa y la cantidad de capas

# In[10]:


# Ajuste en la clase del modelo
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
        x = x.view(x.size(0), -1)  # Aplanar la onda de audio
        x = (x - x.mean()) / x.std()  # Normalización
        for fc in self.fc_layers[:-1]:  # Skip last layer
            x = self.dropout(F.relu(fc(x)))  # ReLU + Dropout
        x = self.fc_layers[-1](x)  # Output layer (no activation)
        return x


# In[11]:


import gc

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


# In[ ]:


input_size = samplerate * 5

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
        model = ExperimentNN(input_size,
                             num_classes,
                             size,
                             layers).to(device)

        # Definir el criterio y optimizador con los pesos de las clases
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Define el scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # Llama a train_model pasándole el scheduler
        validation_loss = train_model(
            model, criterion, optimizer, scheduler=scheduler, epochs=10,
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


# In[ ]:


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


# ### 4. Clasificación

# In[26]:


class CNN(nn.Module):
    def __init__(self, input_channels, num_classes, conv_layers_config):
        super(CNN, self).__init__()

        # Initialize the list to hold the convolutional layers
        self.conv_layers = nn.ModuleList()

        # Initialize the number of input channels for the first layer
        in_channels = input_channels

        # Dynamically create convolutional layers based on the configuration
        for (out_channels, kernel_size, stride, padding) in conv_layers_config:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            in_channels = out_channels  # Update in_channels for the next layer

        # Calculate the size after convolution and pooling to define the fully connected layer
        # Assuming pooling reduces the size by a factor of 2 at each layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Get the final feature map size (after all conv and pooling layers)
        self.final_feature_map_size = self._get_conv_output_size(input_channels, conv_layers_config)
        
        # Define 9 fully connected layers with 256 nodes each
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(self.final_feature_map_size, 256))  # First fully connected layer
        for _ in range(8):  # Add 8 more fully connected layers with 256 nodes
            self.fc_layers.append(nn.Linear(256, 256))
        
        # Output layer
        self.fc_out = nn.Linear(256, num_classes)  # Output layer for classification
        
    def _get_conv_output_size(self, input_channels, conv_layers_config):
        # Sample input size (height x width) to calculate the final feature map size
        # You can adjust these values based on your actual input size
        height = 201  # Replace with your actual input height
        width = 552   # Replace with your actual input width
        
        # Apply each convolutional and pooling layer
        for (out_channels, kernel_size, stride, padding) in conv_layers_config:
            height = (height + 2 * padding - kernel_size) // stride + 1
            width = (width + 2 * padding - kernel_size) // stride + 1
            height = height // 2  # Max pooling halves the height
            width = width // 2    # Max pooling halves the width
        
        # Return the total number of features after all convolutional and pooling layers
        return out_channels * height * width

    def forward(self, x):
        # Apply each convolutional layer followed by ReLU and pooling
        for conv_layer in self.conv_layers:
            x = F.relu(conv_layer(x))
            x = self.pool(x)
        
        # Flatten the output before passing it to the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the feature map

        # Apply the fully connected layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        
        # Output layer (classification)
        x = self.fc_out(x)

        return x


# In[34]:


def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device="cpu"):

    model.train()  # Set the model to training mode
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"\nStarting Epoch {epoch+1}/{num_epochs}")
        
        # Training loop
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            running_loss += loss.item()
            
            # Print loss for every batch
            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                print(f"  Batch {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        # Average loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Training completed. Average Loss: {avg_train_loss:.4f}")
        
    print("\nTraining complete.")

def test_model_configuration(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0  # Total number of samples processed

    # Ensure no gradient computation during evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:  # Assuming test_loader is the correct DataLoader for your test set
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Get the predicted class labels (the one with the highest logit)
            _, predicted = torch.max(outputs.data, 1)

            # Accumulate total samples
            total += labels.size(0)

            # Accumulate correct predictions
            correct += (predicted == labels).sum().item()

    # Calculate average test loss
    test_loss /= len(test_loader)

    # Calculate accuracy
    accuracy = 100 * correct / total

    return test_loss, accuracy

def test_multiple_configurations(train_loader, test_loader, criterion, device, num_epochs=10):
    """
    Test multiple model configurations and evaluate their performance after training.
    """
    # Different configurations for the CNN model
    configurations = [
        # Example of convolutional layers configuration (out_channels, kernel_size, stride, padding)
        [(32, 3, 1, 1), (64, 3, 1, 1), (128, 3, 1, 1)],  # Configuration 1
        [(32, 5, 1, 2), (64, 5, 1, 2)],                   # Configuration 2
        [(16, 3, 1, 1), (32, 3, 1, 1), (64, 3, 1, 1)],   # Configuration 3
        [(64, 3, 1, 1), (128, 3, 1, 1), (256, 3, 1, 1)]  # Configuration 4
    ]
    
    best_model = None
    best_accuracy = 0.0
    
    for idx, conv_layers_config in enumerate(configurations):
        print(f"\nTesting Configuration {idx + 1} with convolutional layers: {conv_layers_config}")
        
        # Initialize the model with the current configuration
        model = CNN(input_channels=1, num_classes=10, conv_layers_config=conv_layers_config)
        model.to(device)  # Send the model to the appropriate device (GPU/CPU)
        
        # Initialize optimizer (e.g., Adam) and train the model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train the model on the training set
        train_model(model, train_loader, criterion, optimizer, num_epochs, device)
        
        # Evaluate the model on the test set
        test_loss, accuracy = test_model_configuration(model, test_loader, criterion, device)

        # Print the results for the current configuration
        print(f"Configuration {idx + 1} Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        
        # Save the best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print(f"\nBest Model Test Accuracy: {best_accuracy:.2f}%")
    return best_model


# In[ ]:


# Assume 'test_loader' is the DataLoader for your test set, and 'criterion' is the loss function (e.g., CrossEntropyLoss)
best_model = test_multiple_configurations(train_spectogram, val_spectogram, criterion, device, 10)


# In[ ]:


test_loss, accuracy = test_model_configuration(best_model, test_spectogram,criterion,device)

print(f"Loss: {test_loss} - Accuracy: {accuracy}")

