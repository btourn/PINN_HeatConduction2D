
from ImportPackages import *
from pytorch_lightning.callbacks import Callback

from tempfile import TemporaryDirectory
import ray
import ray.train
from ray.train import Checkpoint


class PrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("*****Training is starting*****")

    def on_train_end(self, trainer, pl_module):
        print("*****Training is ending*****")

    def on_fit_start(self, trainer, pl_module):
        print("*****Fit step start*****")
    
    def on_fit_end(self, trainer, pl_module):
        print("*****Fit step end*****")

    """
    def on_validation_start(self, trainer, pl_module):
        print(print("*****Validation step start*****"))
    
    def on_validation_end(self, trainer, pl_module):
        print(print("*****Validation step end*****"))"""

    def on_test_start(self, trainer, pl_module):
        print(print("*****Test step start*****"))
    
    def on_test_end(self, trainer, pl_module):
        print(print("*****Test step end*****"))

    def on_predict_start(self, trainer, pl_module):
        print("*****Prediction is strating*****")
        
    def on_predict_start(self, trainer, pl_module):
        print("*****Prediction is ending*****")
        

class MetricTracker(Callback):
    def __init__(self):
        self.collectionTrain = []
        self.collectionValidation = []
        self.collectionTest = []

    def on_validation_epoch_end(self, trainer, module):
        elogs = trainer.logged_metrics # access it here
        if len(elogs)>0:
            train_keys = module.train_keys
            val_keys = module.val_keys
            if train_keys[0] in elogs:
                values = []
                for key in train_keys:
                    values.append(elogs[key])
                self.collectionTrain.append(dict(zip(train_keys, values)))
            if val_keys[0] in elogs:
                values = []
                for key in val_keys:
                    values.append(elogs[key])
                self.collectionValidation.append(dict(zip(train_keys, values)))

    def on_test_epoch_end(self, trainer, module):
        elogs = trainer.logged_metrics # access it here
        if len(elogs)>0:
            test_keys = module.test_keys
            values = []
            for key in test_keys:
                values.append(elogs[key])
            self.collectionTest.append(dict(zip(test_keys, values)))
            #self.collectionTest.append(elogs)


class CustomRayTrainReportCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        should_checkpoint = trainer.current_epoch % 20 == 0
        CHECKPOINT_NAME = "checkpoint.ckpt"

        with TemporaryDirectory() as tmpdir:
            # Fetch metrics
            metrics = trainer.callback_metrics
            metrics = {k: v.item() for k, v in metrics.items()}

            # Add customized metrics
            metrics["epoch"] = trainer.current_epoch

            checkpoint = None
            global_rank = ray.train.get_context().get_world_rank()
            if global_rank==0 and should_checkpoint:
                # Save model checkpoint file to tmpdir
                ckpt_path = os.path.join(tmpdir, CHECKPOINT_NAME)
                trainer.save_checkpoint(ckpt_path, weights_only=False)
                checkpoint = Checkpoint.from_directory(tmpdir)

            # Report to train session
            ray.train.report(metrics=metrics, checkpoint=checkpoint)