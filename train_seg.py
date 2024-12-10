import torch
from argparse import ArgumentParser
from datasets.segdataset import DynnetDataModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from models.seg import SegModel

def main(args):

    # Define the number of classes, backbone, and segmentation model type
    num_classes = args.num_classes
    backbone_type = args.backbone_type
    segmodel_type = args.segmodel_type  # Replace with your segmentation model type
    learning_rate = args.lr

    # Paths to your dataset splits
    train_split = args.train_split
    val_split = args.val_split
    test_split = args.test_split

    # Data module parameters
    batch_size = args.batch_size
    crop_size = args.crop_size
    num_workers = args.num_workers

    # Instantiate the model and data module
    model = SegModel(num_classes=num_classes, backbone_type=backbone_type, segmodel_type=segmodel_type, learning_rate=learning_rate)
    data_module = DynnetDataModule(train_split=train_split, val_split=val_split, test_split=test_split,
                                    batch_size=batch_size, crop_size=crop_size, num_workers=num_workers)
                
    # Logger: CSV Logger
    logger = CSVLogger(
        save_dir=args.model_dir,  # Base directory for logs
        name=args.exp_name        # Subdirectory for this experiment
    )
    
    # Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.model_dir}/{args.exp_name}/checkpoints",  # Checkpoint directory
        filename="{epoch}-{val_loss:.2f}",  # Filename pattern
        monitor="val_loss",  # Metric to monitor
        save_top_k=3,        # Save the best 3 checkpoints only
        mode="min"           # Minimize the monitored metric
    )
    
    # Define the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=args.epochs,                # Number of epochs
        gpus=1 if torch.cuda.is_available() else 0,  # Use GPU if available
        precision=16,                 # Use mixed precision for faster training (optional)
        log_every_n_steps=10,         # Log every 10 steps
        progress_bar_refresh_rate=30  # Refresh the progress bar every 30 steps
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Optionally, evaluate the model on the validation/test set
    test_results = trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--backbone_type', type=str, default='dinov2')
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--segmodel_type', type=str, default='resnet50')
    parser.add_argument('--train_split', type=str)
    parser.add_argument('--val_split', type=str)
    parser.add_argument('--test_split', type=str)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--model_dir', type=str, default='/home/gridsan/manderson/ovdsat/run/seg')
    parser.add_argument('--exp_name', type=str, default='test0')
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)