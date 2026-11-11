import argparse
import datetime
from option import Option
import os
import torch
import time
import trainer
torch.autograd.set_detect_anomaly(False)
import pc_processor
from pc_processor.models.pmf_net import PMFNet
from pc_processor.models.radiant3d import RADIANT3DNet
class Experiment(object):
    def __init__(self, settings: Option):
        self.settings = settings
        # init gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = self.settings.gpu
        pc_processor.utils.init_distributed_mode(self.settings)
        if self.settings.distributed:
            torch.distributed.barrier()

        # set random seed
        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed(self.settings.seed)
        torch.cuda.set_device(int(self.settings.gpu))
        torch.backends.cudnn.benchmark = False

        # init checkpoint
        if not self.settings.distributed or (self.settings.rank == 0):
            self.recorder = pc_processor.checkpoint.Recorder(
                self.settings, self.settings.save_path)
        else:
            self.recorder = None

        self.epoch_start = 0
        # init model
        ModelClass = RADIANT3DNet
        self.model = ModelClass(
            pcd_channels=5,
            img_channels=3,
            nclasses=self.settings.nclasses,
            base_channels=self.settings.base_channels,
            image_backbone=self.settings.img_backbone,
            imagenet_pretrained=self.settings.imagenet_pretrained
        )

        # init trainer
        self.trainer = trainer.Trainer(
            self.settings, self.model, self.recorder)
        # load checkpoint
        self._loadCheckpoint()

    def _loadCheckpoint(self):
        assert self.settings.pretrained_model is None or self.settings.checkpoint is None, "cannot use pretrained weight and checkpoint at the same time"
        if self.settings.pretrained_model is not None:
            if not os.path.isfile(self.settings.pretrained_model):
                raise FileNotFoundError("pretrained model not found: {}".format(
                    self.settings.pretrained_model))
            state_dict = torch.load(
                self.settings.pretrained_model, map_location="cpu")
            incompatible = self.model.load_state_dict(state_dict, strict=False)

            expected_missing = ["lidar_stream.reliability_gamma"]
            if sorted(incompatible.missing_keys) != expected_missing:
                raise RuntimeError(
                    "Unexpected missing checkpoint keys: {}".format(
                        incompatible.missing_keys
                    )
                )
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    "Unexpected checkpoint keys: {}".format(
                        incompatible.unexpected_keys
                    )
                )

            if self.recorder is not None:
                self.recorder.logger.info(
                    "loading pretrained weight from: {}".format(self.settings.pretrained_model))

        if self.settings.checkpoint is not None:
            if not os.path.isfile(self.settings.checkpoint):
                raise FileNotFoundError(
                    "checkpoint file not found: {}".format(self.settings.checkpoint))
            checkpoint_data = torch.load(
                self.settings.checkpoint, map_location="cpu")
            if "model" in checkpoint_data:
                self.model.load_state_dict(checkpoint_data["model"])
                self.trainer.optimizer.load_state_dict(
                    checkpoint_data["optimizer"])
                self.trainer.aux_optimizer.load_state_dict(
                    checkpoint_data["aux_optimizer"])
                if "scheduler" in checkpoint_data:
                    self.trainer.scheduler.load_state_dict(
                        checkpoint_data["scheduler"])
                if "aux_scheduler" in checkpoint_data:
                    self.trainer.aux_scheduler.load_state_dict(
                        checkpoint_data["aux_scheduler"])

                # Legacy checkpoints do not contain scheduler states.
                # Reconstruct scheduler progress using completed batches.
                if "scheduler" not in checkpoint_data:
                    completed_epochs = checkpoint_data["epoch"] + 1
                    completed_steps = completed_epochs * len(self.trainer.train_loader)
                    for _ in range(completed_steps):
                        self.trainer.scheduler.step()
                if "aux_scheduler" not in checkpoint_data:
                    completed_epochs = checkpoint_data["epoch"] + 1
                    completed_steps = completed_epochs * len(self.trainer.train_loader)
                    for _ in range(completed_steps):
                        self.trainer.aux_scheduler.step()

                self.epoch_start = checkpoint_data["epoch"] + 1
            else:
                self.model.load_state_dict(checkpoint_data)
    def run(self):
        t_start = time.time()
        if self.settings.val_only:
            self.trainer.run(0, mode="Validation")
            return
        
        best_val_result = None
        # self.epoch_start = 1
        # update lr after backward (required by pytorch)
        # self.trainer.scheduler.step(self.epoch_start*len(self.trainer.train_loader))
        # if self.settings.optimizer == "Hybrid":
        #     self.trainer.aux_scheduler.step(self.epoch_start*len(self.trainer.train_loader))

        for epoch in range(self.epoch_start, self.settings.n_epochs):
            
            self.trainer.run(epoch, mode="Train")
            if epoch % self.settings.val_frequency == 0 or epoch == self.settings.n_epochs-1:
                val_result = self.trainer.run(epoch, mode="Validation")

                if self.recorder is not None:
                    if best_val_result is None:
                        best_val_result = val_result
                    for k, v in val_result.items():
                        if v >= best_val_result[k]:
                            self.recorder.logger.info(
                                "get better {} model: {}".format(k, v))
                            saved_path = os.path.join(
                                self.recorder.checkpoint_path, "best_{}_model.pth".format(k))
                            best_val_result[k] = v
                            torch.save(self.model.state_dict(), saved_path)

            # save checkpoint
            if self.recorder is not None:
                saved_path = os.path.join(
                    self.recorder.checkpoint_path, "checkpoint.pth")
                checkpoint_data = {
                    "model": self.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "aux_optimizer": self.trainer.aux_optimizer.state_dict(),
                    "epoch": epoch,
                }

                temp_path = saved_path + ".tmp"
                torch.save(checkpoint_data, temp_path)
                os.replace(temp_path, saved_path)
                # log
                if best_val_result is not None:
                    log_str = ">>> Best Result: "
                    for k, v in best_val_result.items():
                        log_str += "{}: {} ".format(k, v)
                    self.recorder.logger.info(log_str)
        cost_time = time.time() - t_start
        if self.recorder is not None:
            self.recorder.logger.info("==== total cost time: {}".format(
                datetime.timedelta(seconds=cost_time)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Options")
    parser.add_argument("config_path", type=str, metavar="config_path",
                        help="path of config file, type: string")
    parser.add_argument("--id", type=int, metavar="experiment_id", required=False,
                        help="id of experiment", default=0)
    args = parser.parse_args()
    exp = Experiment(Option(args.config_path))
    print("===init env success===")
    exp.run()
