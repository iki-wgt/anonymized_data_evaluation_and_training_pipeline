import os
import argparse
import ast
from ultralytics import YOLO

class YOLOv10Trainer:
    def __init__(self, model_path=None, untrained_model='yolov10n.yaml', data_config=None,
        epochs=50, batch_size=16, img_size=640,
        optimizer='auto', momentum=0.937, weight_decay=0.0005, freeze_idx=0,
        lr0=0.01, lrf=0.01, warmup_epochs=3, warmup_momentum=0.8):
        """
        Initialize the YOLOv10Trainer.

        Args:
        - model_path: Path to the pretrained model (.pt file) or None for an untrained model.
        - data_config: Path to the data configuration file.
        - epochs: Number of training epochs.
        - batch_size: Size of each training batch.
        - img_size: Image size for training.
        - untrained_model: Model size configuration (e.g., 'yolov10.yaml', 'yolov10n.yaml', etc.)
        - freeze_idx: Index of layeres which should be frozen. Set [] for no freezing.

        """
        self.model_path = model_path
        self.data_config = data_config
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.untrained_model = untrained_model
        self.is_pretrained = None
        self.model = self.load_model()
        self.opt = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.freeze_idx = freeze_idx
        self.lr0=lr0
        self.lrf=lrf
        self.warmup_epochs=warmup_epochs
        self.warmup_momentum=warmup_momentum

    def load_model(self):
        """
        Load the YOLOv10 model.

        :return: Loaded YOLOv10 model.
        """
        # in case of inerrupted training - set path to last model:
        #return YOLO("/data_private/ss24_weiss_privacy_dataset/src/train_YOLOv10/runs/detect/train/weights/last.pt")

        if self.model_path is not None:
            print(f"Loading pretrained model from {self.model_path}")
            self.is_pretrained = True
            return YOLO(self.model_path)
        else:
            print(f"Creating a new untrained YOLOv10 model using configuration: {self.untrained_model}")
            self.is_pretrained = False
            return YOLO(self.untrained_model)  # Adjust if using a different configuration

    def train(self):
        """
        Train the YOLOv10 model.
        """         

        #Auto-Tuner
        #print(f"Starting training for {self.epochs} epochs with batch size {self.batch_size}...")
        #self.model.train(resume=True) # in case of inerrupted training
        # self.model.tune(data=self.data_config, #auto-tuner for a model
        #                 pretrained=True,
        #                 epochs=self.epochs,
        #                 iterations=10,
        #                 batch=self.batch_size,
        #                 optimizer=self.opt,
        #                 patience=5,
        #                 plots=True,
        #                 save=True,
        #                 cache=False,
        #                 val=False,
        #                 workers=48,
        #                 freeze=self.freeze_idx if len(self.freeze_idx)!=0 else None)
        
        self.model.train(data=self.data_config,
                         pretrained=self.is_pretrained,
                         epochs=self.epochs,
                         batch=self.batch_size,
                         imgsz=self.img_size,
                         optimizer=self.opt,
                         momentum=self.momentum,
                         weight_decay=self.weight_decay,
                         workers=48,
                         device=0,
                         patience=100,
                         save_period=1,
                         plots=True,
                         cache=False,
                         val=False,
                         freeze=self.freeze_idx if len(self.freeze_idx)!=0 else None,
                         lr0=self.lr0,
                         lrf=self.lrf,
                         warmup_epochs=self.warmup_epochs,
                         warmup_momentum=self.warmup_momentum)
        print("Training completed.")

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv10 models with configurable settings.")
    parser.add_argument("data_config", type=str, help="Path to the data configuration file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the pretrained model or None")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Size of each training batch")
    parser.add_argument("--img_size", type=int, default=640, help="Image size for training")
    parser.add_argument("--untrained_model", type=str, default='yolov10n.yaml', help="Untrained model size configuration and path")
    parser.add_argument("--optimizer", type=str, default='auto', help="Optimizer to use")
    parser.add_argument("--momentum", type=float, default=0.937, help="Optimizer momentum")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay rate")
    parser.add_argument("--idx_to_freeze", type=str, default='[]', help="Indices of layers to freeze")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Number of epochs for learning rate warmup, gradually increasing the learning rate from a low value to the initial learning rate to stabilize training early on.")
    parser.add_argument("--warmup_momentum", type=float, default=0.8, help="Initial momentum for warmup phase, gradually adjusting to the set momentum over the warmup period.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Convert the string argument into a Python list
    try:
        idx_to_freeze = ast.literal_eval(args.idx_to_freeze)
    except:
        print("Invalid format for --idx_to_freeze. Please provide a valid list, e.g., '[]' or '[4, 5, 6]'.")
        print("Using default: []")
        idx_to_freeze = []

    # Check for freeze
    if len(idx_to_freeze)==0:
        print("No layer freezing")
    else:
        print(f"Freezing layers: {idx_to_freeze}")

    # Initialize the trainer with command line parameters
    trainer = YOLOv10Trainer(model_path=args.model_path,
                             data_config=args.data_config,
                             epochs=args.epochs,
                             batch_size=args.batch_size,
                             img_size=args.img_size,
                             untrained_model=args.untrained_model,
                             optimizer=args.optimizer,
                             momentum=args.momentum,
                             weight_decay=args.weight_decay,
                             freeze_idx=idx_to_freeze,
                             lr0=args.lr0,
                             lrf=args.lrf,
                             warmup_epochs=args.warmup_epochs,
                             warmup_momentum=args.warmup_momentum)

    # Start training
    trainer.train()
