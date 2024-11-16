from models.seg import CustomSeg
from datasets.segdataset import SegData
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

def main(args):

    loss = smp.losses.DiceLoss(mode='multiclass')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Example dataset (replace with your own)
    train_data = SegData(args.num_classes, args.backbone_type, args.segmodel_type)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_data = SegData()
    train_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    
    '''
    model = CustomSeg(
        num_classes=args.num_classes,
        backbone_type=args.backbone_type,
        segmodel_type=args.segmodel_type
    )

    # Training loop
    num_epochs = args.epochs

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
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

        # val
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                outputs = model(images.cuda())
                # Use thresholding if 'sigmoid' activation was used
                preds = (outputs > 0.5).float()
        '''

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--backbone_type', type=str, default='dinov2')
    parser.add_argument('--segmodel_type', type=str, default='resnet50')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--model_dir', type=str, default='/home/gridsan/manderson/ovdsat/run/seg')
    parser.add_argument('--exp_name', type=str, default='test0')
    #parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)