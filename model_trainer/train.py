import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
from model import SongModel
import os

def train_model():
    # Load data
    print("Loading dataset...")
    df = pd.read_csv("segments.csv")
    
    # Extract features and labels
    feature_cols = [col for col in df.columns if col.startswith("feat_")]
    X = df[feature_cols].values
    y = df["label"].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Feature columns: {feature_cols}")
    print(f"Label distribution:\n{pd.Series(y).value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Create data loaders
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    # Build model
    model = SongModel(input_size=27)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training parameters
    num_epochs = 100
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print(f"\nTraining model for {num_epochs} epochs...")
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            val_preds = (val_outputs > 0.5).float()
            val_acc = (val_preds == y_test_tensor).float().mean()
        
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'feature_cols': feature_cols,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc.item()
            }, 'best_song_model.pth')
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    checkpoint = torch.load('best_song_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_test_tensor)
        final_preds = (final_outputs > 0.5).float().squeeze()
        
        print(f"\n✅ Final Results:")
        print(f"Validation Accuracy: {checkpoint['val_acc']:.4f}")
        print(f"Best Validation Loss: {checkpoint['val_loss']:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, final_preds))
        
        # Confusion matrix
        print(f"Confusion Matrix:")
        print(confusion_matrix(y_test, final_preds))
    
    print(f"\n✅ Model saved to 'best_song_model.pth'")
    return model, scaler, feature_cols


if __name__ == "__main__":
    # Train the model
    model, scaler, feature_cols = train_model()
    
