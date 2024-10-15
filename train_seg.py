from models.seg import CustomSeg
from datasets.segdataset import SegData

loss = smp.losses.DiceLoss(mode='multiclass')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

from torch.utils.data import DataLoader

# Example dataset (replace with your own)
train_dataset = SegmentationDataset(images, masks, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Training loop
model.train()
num_epochs = 10

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.cuda(), masks.cuda()

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Eval
model.eval()
with torch.no_grad():
    for images, masks in val_loader:
        outputs = model(images.cuda())
        # Use thresholding if 'sigmoid' activation was used
        preds = (outputs > 0.5).float()
