from data import USPTO_AugmentedDataModule
from models import Seq2SeqTransformer
import os, sys
import config
import pytorch_lightning as pl
import re, string
import tqdm

def compute_predictions(ckpt_path, _type, n_augm, representation, cross=False):
    #print(os.getcwd())
    data_module = USPTO_AugmentedDataModule(
        cfg.data.path_prefix, cfg.data.batch_size,
        _type, n_augm, representation,
        n_workers=0,
        cross_distribution=cross,
        verbose=True
    )
    data_module.setup(stage="predict")

    model = Seq2SeqTransformer.load_from_checkpoint(ckpt_path,
        cfg=cfg,
        data_module=data_module,
        learning_rate=cfg.training.lr,
        num_encoder_layers=cfg.model.transformer.encoder_depth, num_decoder_layers=cfg.model.transformer.decoder_depth,
        emb_size=cfg.model.transformer.embedding_size, nhead=cfg.model.transformer.attn_heads,
        vocab_size=len(data_module.vocabulary),
        padding_idx=data_module.vocabulary["<pad>"], start_idx=data_module.vocabulary["<start>"],
        end_idx=data_module.vocabulary["<end>"],
        max_length=cfg.model.max_length,
        dim_feedforward=cfg.model.transformer.d_ff
    )

    print("MODEL DEVICE:", model.device)

    #cfg.device = config.best_gpus(1)
    cfg.name = model_name
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.devices,
        auto_select_gpus=False,
        enable_checkpointing=False #,
        #limit_predict_batches=1
    )

    if cross:
        print("CROSS=True predictions")
        test_preds = trainer.predict(
            model=model,
            dataloaders=[data_module.ts_cross_dataloader]
        )
    else:
        print("CROSS=False predictions")
        test_preds = trainer.predict(
            model=model,
            dataloaders=[data_module.ts_in_dataloader, data_module.ts_out_dataloader]
        )

    # NB: predictions are automatically stored in "final_predictions" folder with on_predict_epoch_end hook


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cfg = config.make_config(config.default_config)

    # *** CONFIG ***
    # set this before running
    CROSS = True # TODO
    _TYPE = 1
    N_AUGM = 1
    REPRESENTATION = "formula"

    #model_name = "TRANS_FINAL_RUN"
    model_name = "TRANS_FINAL_RUN_CROSS"

    #checkpoint_name = "epoch=epoch=20-step=step=285380-val_acc=loss_validation=1.0879.ckpt"

    for n_augm in [1, 5]:
        for _type in [1, 2]:
            for representation in ["formula", "smiles"]:
                name = model_name + "_" + representation + "_type" + str(_type) + "_x" + str(n_augm)
                ckpt_path = os.path.join("lightning_logs", name, "version_0", "checkpoints")
                _, _, files = next(os.walk(ckpt_path))

                ckpt_name = files[0]
                ckpt_path = os.path.join(ckpt_path, ckpt_name)

                print("Checkpoint path:", ckpt_path)
                if True:
                    compute_predictions(ckpt_path, _type, n_augm, representation, cross=CROSS)

    print("ALL GOOD!")
