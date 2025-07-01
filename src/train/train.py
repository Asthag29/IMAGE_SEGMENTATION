# %%
import torch 
from torch import nn
from dataloaders import TrainSegmentationDataloader, TestSegmentationDataloader
from seg_model import FishSegmentation
from tqdm import tqdm 
import matplotlib.pyplot as plt
import typer



# %%
app = typer.Typer()


def train(model , epochs , train_dataloaders , valid_dataloaders, optimizer, loss=None, model_path=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Training on: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")

    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Training started...\n")
    
    for epoch in range(epochs):
        
        # Training Phase
        model.train()
        running_train_loss = 0.0
        
        train_bar = tqdm(
            train_dataloaders, 
            desc=f"Epoch {epoch+1}/{epochs} - Training",
            leave=True 
        )
        
        for batch_idx, (image, mask) in enumerate(train_bar):
            image, mask = image.to(device), mask.to(device)

            optimizer.zero_grad()
            predicted = model(image)

            loss= nn.BCEWithLogitsLoss()(predicted,mask)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            running_train_loss += batch_loss
            
            # Update display
            train_bar.set_postfix({
                'Loss': f'{running_train_loss:.4f}',
            
            })
        
        avg_train_loss = running_train_loss / len(train_dataloaders)
        train_losses.append(avg_train_loss)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        
        val_bar = tqdm(
            valid_dataloaders, 
            desc=f"Epoch {epoch+1}/{epochs} - Validation",
            leave=False
        )
        
        with torch.no_grad():
            for batch_idx, (image_valid, mask_valid) in enumerate(val_bar):
                image_valid, mask_valid = image_valid.to(device), mask_valid.to(device)
            
                predicted = model(image_valid)
                loss = nn.BCEWithLogitsLoss()(predicted, mask_valid)
                running_val_loss += loss.item()
                
                val_bar.set_postfix({
                    'Loss': f'{running_val_loss:.4f}',
                    
                })
        
        avg_val_loss = running_val_loss / len(valid_dataloaders)
        val_losses.append(avg_val_loss)
    
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
       
        model_path= "../../src/model_paths/best_fish_segmentation_model.pth"
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, model_path)
            print(f'✅ Best model saved! Val Loss: {avg_val_loss:.4f}')
        
        print('-' * 50)
    
    print(f"\n🎉 Training completed! Best validation loss: {best_val_loss:.4f}")
    return train_losses, val_losses


# %%

def test_model(test_dataloader, model_path='../../src/model_paths/best_fish_segmentation_model.pth',save_path='../../src/reports/fish-test.png'):
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FishSegmentation()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Get one batch
    images = next(iter(test_dataloader))
    images = images.to(device)
    
    # Predict
    with torch.no_grad():
        predictions = model(images)
        
    # Plot first 4 images
    _, axes = plt.subplots(2, 4, figsize=(15, 8))
    
    for i in range(min(4, len(images))):
        # Original image
        img = images[i].cpu().squeeze()
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Original Image {i+1}')
        axes[0, i].axis('off')
        
        # Predicted mask
        # Predicted mask
        mask = predictions[i].cpu().squeeze()
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'Predicted Mask {i+1}')
        axes[1, i].axis('off')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


# %%

def plot_losses(train_losses, val_losses, save_path='../../src/reports/loss_graph.png', title="Training and Validation Loss"):
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Find and mark the minimum validation loss
    min_val_idx = val_losses.index(min(val_losses))
    plt.annotate(f'Best Val Loss: {min(val_losses):.4f}', 
                xy=(min_val_idx + 1, min(val_losses)), 
                xytext=(min_val_idx + 1, min(val_losses) + 0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Loss plot saved at: {save_path}")
    plt.show()


# %%
@app.command()
def train_cli(
    epochs: int = typer.Option(1, "--epochs", "-e", help="Number of training epochs"),
    lr: float = typer.Option(0.001, "--learning-rate", "-lr", help="Learning rate")
):
    
    # Initialize model and optimizer
    model = FishSegmentation()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    traindl, validdl = TrainSegmentationDataloader()
    test_dataloader= TestSegmentationDataloader()
    model_path = "../../src/model_paths/best_fish_segmentation_model.pth"

    print(f"🚀 Starting training with:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    
    
    # Train the model using existing function
    train_losses, val_losses = train(model, epochs, traindl, validdl, optimizer)
    
    # Load the saved model to get best epoch info
    checkpoint = torch.load(model_path)
    best_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']

    # visulaizing the graph
    plot_losses(train_losses, val_losses, title="Training and Validation Loss")
    
    # visualizing the testing data
    test_model(test_dataloader, model_path=model_path)
    
    print(f"\n📊 Training Summary:")
    print(f"📁 Model saved at: {model_path}")
    print(f"🏆 Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"📊 Loss graph saved at: loss_graph.png")
    print(f"🐟 Fish test images saved at: fish_test.png")



if __name__ == "__main__":
    app() 


