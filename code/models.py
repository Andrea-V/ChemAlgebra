import os, sys
import torch
from torch import nn
import torch.nn.functional as F
import math
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import Tensor
import tqdm
import data
import statistics as stat
#from pytorch_beam_search import seq2seq
from torch.optim.swa_utils import SWALR
from search_algorithms import beam_search

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, padding_idx, bidirectional):
        super(EncoderRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.gru = nn.GRU(
            hidden_size, hidden_size,
            batch_first=True, num_layers=n_layers,
            bidirectional=bidirectional
        )

    def forward(self, x, h_0=None):
        embedded = self.embedding(x)
        output = embedded
        output, h_n = self.gru(output, h_0)
        return output, h_n

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, padding_idx):
        super(DecoderRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.gru = nn.GRU(
            hidden_size, hidden_size,
            batch_first=True, num_layers=n_layers
        )
        self.readout = nn.Linear(hidden_size, vocab_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, encoder_outputs=None, h_0=None):
        output = self.embedding(input)
        output = F.relu(output)
        output, h_n = self.gru(output, h_0)
        # output = self.softmax(self.out(output))
        output = self.readout(output)
        return output, h_n, None

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, attn_heads, n_layers, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, attn_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=n_layers)
        self.readout = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, encoder_outputs, h_0=None):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        # similar to encoder-decoder attention in transformers
        attn_output, attn_weights = self.attn(embedded, encoder_outputs, encoder_outputs)

        output = F.relu(attn_output)
        output, h_n = self.gru(output, h_0)

        output = self.readout(output)
        return output, h_n, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class Seq2SeqGRU(pl.LightningModule):
    def __init__(
            self, learning_rate,
            vocab_size, hidden_size, depth, use_attention, attn_heads,
            padding_idx, start_idx, end_idx, max_length, bidirectional
    ):
        super(Seq2SeqGRU, self).__init__()
        self.encoder = EncoderRNN(vocab_size, hidden_size, padding_idx=padding_idx, n_layers=depth,
                                  bidirectional=bidirectional)

        self.bidirectional = bidirectional
        self.vocab_size = vocab_size
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.depth = depth

        if self.bidirectional:
            decoder_hidden_size = 2 * hidden_size
        else:
            decoder_hidden_size = hidden_size

        if use_attention:
            self.decoder = AttnDecoderRNN(vocab_size, decoder_hidden_size, attn_heads, depth)
        else:
            self.decoder = DecoderRNN(vocab_size, decoder_hidden_size, padding_idx=padding_idx, n_layers=depth)


    def forward(self, x, y):
        y_input = y[:, :-1]
        y_target = y[:, 1:]

        encoder_outputs, h_n = self.encoder(x)

        if self.bidirectional:
            batch_size = h_n.shape[1]
            h_n = h_n.reshape((self.depth, batch_size, -1))

        y_hat, _, _ = self.decoder(y_input, encoder_outputs, h_n)
        y_target = F.one_hot(y_target, num_classes=self.vocab_size).float()

        return y_hat, y_target

    # greedy decoding
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        def predict_single_sample(x, bidirectional):
            # we want to do decoding for single sample only
            assert x.shape[0] == 1

            enc_output, enc_h_n = self.encoder(x)
            # store here decoded indices and attentions weights
            decoded = [self.start_idx]
            # decoder_attentions = torch.zeros(self.max_length, x.shape[1], device=self.device, dtype=torch.float)

            y_input_i = torch.ones((1, 1), device=self.device, dtype=torch.int64) * self.start_idx

            if bidirectional:
                batch_size = enc_h_n.shape[1]
                dec_h_i = enc_h_n.reshape((self.depth, batch_size, -1))
            else:
                dec_h_i = enc_h_n

            for di in range(self.max_length):
                output, dec_h_i, attn_weights = self.decoder(y_input_i, enc_output, dec_h_i)

                # if attn_weights is not None:
                #    decoder_attentions[di, :] = attn_weights.squeeze().detach().data

                topv, topi = output.squeeze().detach().data.topk(1)
                decoded.append(topi.squeeze().detach().item())

                if topi.item() == self.end_idx:
                    break

                y_input_i = topi.view(1, 1).detach()

            return decoded  # , decoder_attentions[:di + 1, :]

        y_pred = []
        x, y, x_lens, y_lens = batch
        for x_i, y_i, x_len, y_len in zip(x, y, x_lens, y_lens):
            # re-insert batch dimension (because model needs it)
            x_i = x_i.view(1, -1)
            y_pred.append(predict_single_sample(x_i, self.bidirectional))

        return y_pred

    def training_step(self, batch, batch_idx):
        x, y, x_lens, y_lens = batch
        y_hat, y_target = self(x, y)
        loss = F.cross_entropy(y_hat, y_target)
        #self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, x_lens, y_lens = batch
        y_hat, y_target = self(x, y)
        loss = F.cross_entropy(y_hat, y_target)
        #self.log("validation_loss", loss)
        return loss

    # def test_step(self, batch, batch_idx):
    #     x, y, x_lens, y_lens = batch
    #     y_hat, y_target = self(x, y)
    #     loss = F.cross_entropy(y_hat, y_target)
    #     #self.log("test_loss", loss, on_epoch=True, on_step=False)
    #     return loss

    def training_epoch_end(self, outputs):
        outputs = [ out["loss"].detach().item() for out in outputs ]

        self.log('step', float(self.trainer.current_epoch))
        self.log("loss/training", stat.mean(outputs))

    def validation_epoch_end(self, outputs):
        outputs = [ out.detach().item() for out in outputs ]

        self.log('step', float(self.trainer.current_epoch))
        self.log("loss/validation", stat.mean(outputs))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(pl.LightningModule):
    def __init__(self, cfg, data_module, learning_rate,
                 num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, nhead: int, vocab_size: int,
                 padding_idx, start_idx, end_idx, max_length,
                 dim_feedforward: int = 512, dropout: float = 0.1
                 ):
        super(Seq2SeqTransformer, self).__init__()
        self.cfg = cfg
        self.data_module = data_module
        self.lr = learning_rate
        self.transformer = nn.Transformer(
            d_model=emb_size, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.vocab_size = vocab_size
        self.generator = nn.Linear(emb_size, vocab_size)
        self.src_tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.max_length = max_length

        self.name = self.cfg.name

        self.save_hyperparameters()

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        embeddings = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer.decoder(embeddings, memory, tgt_mask)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(torch.bool)

        src_padding_mask = (src == self.padding_idx).transpose(0, 1)
        tgt_padding_mask = (tgt == self.padding_idx).transpose(0, 1)

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, x, y):

        # the transformer expects inputs to be sequence first
        # need to swap dimensions
        x = torch.transpose(x, 0, 1)
        y_input = torch.transpose(y, 0, 1)

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(x, y_input)

        src_emb = self.positional_encoding(self.src_tok_emb(x))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(y_input))
        outs = self.transformer(
            src_emb, tgt_emb, src_mask, tgt_mask, None,
            src_padding_mask, tgt_padding_mask, src_padding_mask
        )
        y_pred = self.generator(outs)

        # swap back to batch-first representation before doing the rest of processing
        y_pred = torch.transpose(y_pred, 0, 1)


        return y_pred

    # function to generate output sequence using beam search
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, x_lens, y_lens = batch

        y_pred, log_probs = beam_search(
            self, x,
            sos_token=self.start_idx,
            predictions=self.max_length,
            beam_width = self.cfg.model.transformer.beam_width,
            batch_size = x.shape[0],
            progress_bar = False
        )
        return y_pred

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx):

        return

        assert dataloader_idx in [0, 1, 2]

        predictions_stores = [
            self.predictions_store["in"],
            self.predictions_store["x"],
            self.predictions_store["out"]
        ]

        predictions_stores[dataloader_idx].append(outputs)


    def training_step(self, batch, batch_idx):
        x, y, x_lens, y_lens = batch
        y_target = y[:, 1:] # F.one_hot(y[:, 1:], num_classes=self.vocab_size).float()
        y_input = y[:, :-1]

        y_pred = self(x, y_input)

        # alternativa: fare softmax a class probabilities di target (one-hot)
        y_pred = y_pred.permute((0, 2, 1))

        loss = F.cross_entropy(y_pred, y_target, label_smoothing=self.cfg.loss.label_smoothing)
        self.log("loss/training", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, x_lens, y_lens = batch
        y_target = y[:, 1:] # F.one_hot(y[:, 1:], num_classes=self.vocab_size).float()
        y_input = y[:, :-1]

        y_pred = self(x, y_input)

        # alternativa: fare softmax a class probabilities di target (one-hot)
        y_pred = y_pred.permute((0, 2, 1))

        loss = F.cross_entropy(y_pred, y_target, label_smoothing=self.cfg.loss.label_smoothing)
        self.log("loss/validation", loss)
        self.log("loss_validation", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, x_lens, y_lens = batch
        y_target = y[:, 1:] # F.one_hot(y[:, 1:], num_classes=self.vocab_size).float()
        y_input = y[:, :-1]

        y_pred = self(x, y_input)

        # alternativa: fare softmax a class probabilities di target (one-hot)
        y_pred = y_pred.permute((0, 2, 1))

        loss = F.cross_entropy(y_pred, y_target, label_smoothing=self.cfg.loss.label_smoothing)
        self.log("loss/test", loss)
        return loss

    def on_predict_epoch_end(self, results):

        if self.cfg.data.cross:
            store_final_predictions(
                results[0],
                self.data_module.vocabulary,
                self.cfg.eval.final_preds_path, self.cfg.name + "_on_" + self.data_module.dataset_name, "test_cross"
            )
        else:
            store_final_predictions(
                results[0],
                self.data_module.vocabulary,
                self.cfg.eval.final_preds_path, self.cfg.name + "_on_" + self.data_module.dataset_name, "test_in"
            )
            store_final_predictions(
                results[1],
                self.data_module.vocabulary,
                self.cfg.eval.final_preds_path, self.cfg.name + "_on_" + self.data_module.dataset_name, "test_out"
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        print("\nChosen learning rate:", self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
             optimizer, mode="min", factor=0.5, patience=self.cfg.training.scheduler_patience
        )

        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #    optimizer, T_0=1000, T_mult=1,
        #)
        #scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])

        #scheduler = torch.optim.swa_utils.SWALR(
        #    optimizer, anneal_strategy="linear", anneal_epochs=5, swa_lr=1e-4,
        #)
        #return optimizer
        #return [optimizer], [scheduler]
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": self.cfg.training.val_check_interval,
                "monitor": "loss/validation",
                "strict": True,
                "name": "learning_rate"
            }
        }

def store_final_predictions(predictions, vocab, final_preds_path, dataset_name, subset_name):
    if not os.path.exists(final_preds_path):
        os.makedirs(final_preds_path)

    # store raw predictions
    raw_preds = []
    for batch in predictions:
        batch = batch.view((-1, batch.shape[-1]))

        for pred in batch:
            pred = pred.tolist()
            pred_str = vocab.lookup_tokens(pred)
            pred_str = "".join(pred_str)
            raw_preds.append(pred_str + "\n")

    with open(os.path.join(final_preds_path, dataset_name + "_" + subset_name + "_raw.txt"), "w") as out_fp:
        out_fp.writelines(raw_preds)

    # store polished predictions (without special tokens)
    final_preds = []
    for batch in predictions:
        batch = batch.view((-1, batch.shape[-1]))

        for pred in batch:
            pred = pred.tolist()
            pred_str = vocab.lookup_tokens(pred)
            pred_str = "".join(filter(lambda x: x not in ["<start>", "<end>", "<pad>", "<unk>"], pred_str))
            final_preds.append(pred_str + "\n")

    with open(os.path.join(final_preds_path, dataset_name + "_" + subset_name + "_final.txt"), "w") as out_fp:
        out_fp.writelines(final_preds)
