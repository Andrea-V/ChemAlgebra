from models import Seq2SeqGRU, Seq2SeqTransformer
from data import USPTO_AugmentedDataModule
import os, sys
import torch
import pytorch_lightning as pl
import config
import tqdm
from torch.utils.data import DataLoader

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

pl.seed_everything(23041993, workers=True)

def train_all_models_on_augmented_dataset(cfg, n_augm, _type, represent):

    data_module = USPTO_AugmentedDataModule(
        cfg.data.path_prefix, cfg.data.batch_size,
        _type, n_augm, represent,
        cross_distribution=cfg.data.cross,
        n_workers=cfg.data.n_workers,
        verbose=True
    )

    print("The vocabulary size is:", len(data_module.vocabulary))
    data_module.setup(stage="fit")
    data_module.setup(stage="validate")

    if False:
        model_gru = Seq2SeqGRU(
            learning_rate=cfg.training.lr,
            vocab_size=len(data_module.vocabulary), hidden_size=cfg.model.gru.hidden_size,
            use_attention=False, attn_heads=cfg.model.gru.attn_heads,
            depth=cfg.model.gru.depth, padding_idx=data_module.vocabulary["<pad>"],
            start_idx=data_module.vocabulary["<start>"], end_idx=data_module.vocabulary["<end>"],
            max_length=cfg.model.max_length, bidirectional=True
        )

        model_gru_attn = Seq2SeqGRU(
            learning_rate=cfg.training.lr,
            vocab_size=len(data_module.vocabulary), hidden_size=cfg.model.gru.hidden_size,
            use_attention=True, attn_heads=cfg.model.gru.attn_heads,
            depth=cfg.model.gru.depth, padding_idx=data_module.vocabulary["<pad>"],
            start_idx=data_module.vocabulary["<start>"], end_idx=data_module.vocabulary["<end>"],
            max_length=cfg.model.max_length, bidirectional=True
        )

    model_trans = Seq2SeqTransformer(
        cfg=cfg,
        data_module=data_module,
        learning_rate=cfg.training.lr,
        num_encoder_layers=cfg.model.transformer.encoder_depth, num_decoder_layers=cfg.model.transformer.decoder_depth,
        emb_size=cfg.model.transformer.embedding_size, nhead=cfg.model.transformer.attn_heads,
        vocab_size=len(data_module.vocabulary),
        padding_idx=data_module.vocabulary["<pad>"], start_idx=data_module.vocabulary["<start>"], end_idx=data_module.vocabulary["<end>"],
        max_length=cfg.model.max_length,
        dim_feedforward=cfg.model.transformer.d_ff
    )

    #for model, model_name in zip([model_gru, model_gru_attn, model_trans], ["GRU", "GRU_ATTN", "TRANS"]):
    for model, model_name in zip([model_trans], ["TRANS"]):

        model_name = model_name + "_" + cfg.name

        logger = TensorBoardLogger(
            save_dir="lightning_logs",
            name=model_name + "_" + represent + "_type" + str(_type) + "_x" + str(n_augm)
        )

        trainer = pl.Trainer(
            logger=logger,
            #strategy=DDPPlugin(find_unused_parameters=False),
            callbacks=[
                ModelCheckpoint(
                    save_top_k=1,
                    monitor="loss/validation",
                    filename="epoch={epoch:02d}-step={step}-val_acc={loss_validation:.4f}"
                ),
                EarlyStopping(
                    monitor="loss/validation",
                    mode="min",
                    min_delta=cfg.training.min_delta,
                    patience=cfg.training.early_stopping_patience,
                ),
                LearningRateMonitor(log_momentum=False),
                #StochasticWeightAveraging(
                #    swa_epoch_start=1,
                #    swa_lrs=1e-3,
                #    annealing_epochs=10,
                #    annealing_strategy="linear"
                #)
            ],
            accelerator="gpu",
            devices=cfg.devices,
            max_epochs=cfg.training.max_epochs,
            fast_dev_run=cfg.training.fast_dev_run,
            num_sanity_val_steps=3,
            auto_lr_find=False,
            auto_scale_batch_size=False,
            auto_select_gpus=False,
            enable_checkpointing=True,
            #limit_train_batches=1,
            #limit_val_batches=1,
            val_check_interval=cfg.training.val_check_interval
        )

        trainer.fit(
            model=model,
            datamodule=data_module
        )

        if False:
            # test and predict on all three test sets
            test_preds = trainer.predict(
                model=model,
                dataloaders=[data_module.ts_in_dataloader, data_module.ts_x_dataloader, data_module.ts_out_dataloader]
            )

            # store final predictions
            #for test_pred, name in zip(test_preds, ["test_in", "test_x", "test_out"]):
            #    store_final_predictions(
            #        test_pred, data_module.vocabulary,
            #        cfg.eval.final_preds_path, model_name + "_on_" + data_module.dataset_name, name
            #    )

    return None, None, model_trans


def store_selected_parameters(model, data_module, final_preds_path, dataset_name, subset_name):
    if not os.path.exists(final_preds_path):
        os.makedirs(final_preds_path)

    batch_size = data_module.batch_size
    learning_rate = model.learning_rate

    string_to_write = "Batch_size: " + str(batch_size) + " , learning_rate: " + str(learning_rate) + " .\n"

    with open(os.path.join(final_preds_path, dataset_name + "_" + subset_name + ".txt"), "w") as out_fp:
        out_fp.write(string_to_write)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    n_runtime_errors = 0
    cfg = config.make_config(config.default_config)
    for n_augm in [1, 5]:
        for represent in ["smiles", "formula"]:
            for _type in [1, 2]:
                print("*** NEW TRAINING: type:", _type, ", represent:", represent, ", N:", n_augm, "***")
                try:
                    model_gru, model_gru_attn, model_trans = train_all_models_on_augmented_dataset(cfg, n_augm, _type, represent)
                except RuntimeError as e:
                    print("Runtime error!")
                    print("The error raised is: ", e)
                    n_runtime_errors += 1
                    continue

    print("- n_runtime_errors:", n_runtime_errors)
    print("ALL GOOD!")
