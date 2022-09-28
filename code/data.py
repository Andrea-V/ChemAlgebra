import os.path
import sys
from collections import Counter
import pytorch_lightning as pl
import torch.utils.data
import re
from torch.utils.data import DataLoader
import torchtext.vocab.vocab
from torch import Tensor, tensor
from torch.nn.utils.rnn import pad_sequence
import pickle

class USPTO_AugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset for the USPTO_augmented_* tasks.
    """
    def __init__(self, data_path, split, tokenizer, vocabulary, verbose=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab = vocabulary
        self.data_path = data_path

        assert split in ["train", "valid", "test_in_x1", "test_out_x1", "train_cross", "valid_cross", "test_cross"]

        self.reactants = []
        self.products = []
        self.reactants_lenghts = []
        self.products_lenghts = []

        if verbose:
            print("Loading data...", end="")

        with open(os.path.join(self.data_path, "src-" + split + ".txt"), "r") as src_fp:
            with open(os.path.join(self.data_path, "tgt-" + split + ".txt"), "r") as tgt_fp:
                for line in src_fp.readlines():
                    tokens = self.tokenizer.tokenize(line.strip())
                    self.reactants.append(["<start>"] + tokens + ["<end>"])
                    self.reactants_lenghts.append(len(tokens) + 2)

                for line in tgt_fp.readlines():
                    tokens = self.tokenizer.tokenize(line.strip())
                    self.products.append(["<start>"] + tokens + ["<end>"])
                    self.products_lenghts.append(len(tokens) + 2)

        self.max_reactants_len = max(self.reactants_lenghts)
        self.max_products_len = max(self.products_lenghts)

        if verbose:
            print("done!")
            print("Converting from string to indexes...", sep="")

        self.reactants = [ tensor(self.vocab.lookup_indices(react), dtype=torch.int64) for react in self.reactants ]#[:1000] #TODO TODO
        self.products =  [ tensor(self.vocab.lookup_indices(prod), dtype=torch.int64) for prod in self.products ]#[:1000] # TODO TODO

        if verbose:
            print("done!")

    def __getitem__(self, key):
        return self.reactants[key], self.products[key]

    def __len__(self):
        assert len(self.reactants) == len(self.products)
        return len(self.reactants)


class Tokenizer():
    """
    Creates a tokenizer given the specific regex_pattern.
    a.t.m. only used by USPTO_AugmentedDataset
    """
    def __init__(self, regex_pattern):
        self.pattern = regex_pattern
        self.regex = re.compile(regex_pattern)

    def tokenize(self, input_line):
        tokens = [token for token in self.regex.findall(input_line)]

        try:
            assert input_line == ''.join(tokens)
        except AssertionError:
            print("AssertionError:")
            print("SMI_LINE:", input_line)
            print("TOKENS:", tokens)
            remaining = self.regex.sub("", input_line)
            print("REMAINING:", remaining)
            assert False

        return tokens


class USPTO_AugmentedDataModule(pl.LightningDataModule):
    """
    Data module containing all the required USPTO_Augmented dataset for the different data_splits.
    """
    def __init__(self, path_prefix, batch_size, type, n_augm, represent, n_workers, cross_distribution, verbose=True):
        super().__init__()
        self.data_path = path_prefix
        self.batch_size = batch_size
        self.type = type
        self.n_augm = n_augm
        self.repr = represent
        self.verbose = verbose
        self.n_workers = n_workers
        self.cross_distribution = cross_distribution

        assert type == 1 or type == 2
        assert n_augm == 1 or n_augm == 5 or n_augm == 10
        assert represent == "smiles" or represent == "formula"

        folder_name = "USPTO_augmented_type" + str(type) + "_x" + str(n_augm) + "_" + represent
        self.dataset_name = folder_name
        self.data_path = os.path.join(path_prefix, folder_name)
        vocab_path = os.path.join(self.data_path, "vocab/")

        print("vocab path:", vocab_path)

        if not os.path.exists(vocab_path):
            print("- vocab path not exists!")
            os.makedirs(vocab_path)
            self.tokenizer, self.vocabulary, self.max_seq_len = self._preprocess(represent, self.data_path, vocab_path)
        else:
            print("- vocab path exists!")
            with open(os.path.join(vocab_path, "vocabulary.pkl"), "rb") as vocab_fp:
                self.vocabulary = pickle.load(vocab_fp)
            with open(os.path.join(vocab_path, "tokenizer.pkl"), "rb") as tok_fp:
                self.tokenizer = pickle.load(tok_fp)
            with open(os.path.join(vocab_path, "max_seq_length.pkl"), "rb") as msl_fp:
                self.max_seq_len = pickle.load(msl_fp)

    def _preprocess(self, representation, data_path, vocab_path, verbose=True):
        if representation == "smiles":
            tokenizer = Tokenizer("(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|\{[0-9]+\})")
        else:
            tokenizer = Tokenizer("([A-Z][a-z]?|\.|-[0-9]*|\+[0-9]*|[0-9]+|\{[0-9]+\})")

        vocab_counter = Counter()
        max_len = 0

        if verbose:
            print("Creating a vocabulary for the task...")

        _, _, filenames = next(os.walk(self.data_path))
        for filename in filenames:

            if verbose:
                print("- Preprocessing", filename)

            with open(os.path.join(self.data_path, filename), "r") as src_fp:
                for line in src_fp.readlines():
                    tokens = tokenizer.tokenize(line.strip())
                    vocab_counter.update(tokens)

                    # update len if needed
                    this_len = len(tokens) + 2
                    if max_len < this_len:
                        max_len = this_len

        if verbose:
            print("done!")
            print("Creating the vocabulary...", end="")

        special_tokens = ["<start>", "<end>", "<pad>", "<unk>"]
        vocab = torchtext.vocab.vocab(vocab_counter, specials=special_tokens)
        vocab.set_default_index(vocab["<unk>"])

        if verbose:
            print("done!")

        with open(os.path.join(vocab_path, "vocabulary.pkl"), "wb") as vocab_fp:
            pickle.dump(vocab, vocab_fp)
        with open(os.path.join(vocab_path, "tokenizer.pkl"), "wb") as tok_fp:
            pickle.dump(tokenizer, tok_fp)
        with open(os.path.join(vocab_path, "max_seq_length.pkl"), "wb") as msl_fp:
            pickle.dump(max_len, msl_fp)

        return tokenizer, vocab, max_len

    def setup(self, stage):
        """
            different stages = {fit, validate, test, predict}
        """
        if stage == "fit":
            if self.verbose:
                print("Loading training data...")

            if self.cross_distribution:
                self.tr_data = USPTO_AugmentedDataset(self.data_path, "train_cross", self.tokenizer, self.vocabulary, verbose=False)
            else:
                self.tr_data = USPTO_AugmentedDataset(self.data_path, "train", self.tokenizer, self.vocabulary, verbose=False)

        if stage == "validate":
            if self.verbose:
                print("Loading validation data...")

            if self.cross_distribution:
                self.vl_data = USPTO_AugmentedDataset(self.data_path, "valid_cross", self.tokenizer, self.vocabulary, verbose=False)
            else:
                self.vl_data = USPTO_AugmentedDataset(self.data_path, "valid", self.tokenizer, self.vocabulary, verbose=False)

        if stage == "test" or stage == "predict":
            if self.cross_distribution:
                if self.verbose:
                    print("Loading test (cross) data...")

                self.ts_cross_data = USPTO_AugmentedDataset(self.data_path, "test_cross", self.tokenizer, self.vocabulary, verbose=False)
                self.ts_cross_dataloader = DataLoader(self.ts_cross_data, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, collate_fn=self._pad_collate)

            else:
                if self.verbose:
                    print("Loading test (in) data...")

                self.ts_in_data = USPTO_AugmentedDataset(self.data_path, "test_in_x1", self.tokenizer, self.vocabulary, verbose=False)
                self.ts_in_dataloader = DataLoader(self.ts_in_data, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, collate_fn=self._pad_collate)

                #if self.verbose:
                #    print("Loading test (cross) data...")

                #self.ts_x_data = USPTO_AugmentedDataset(self.data_path, "test_x", self.tokenizer, self.vocabulary, verbose=False)
                #self.ts_x_dataloader = DataLoader(self.ts_x_data, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, collate_fn=self._pad_collate)

                if self.verbose:
                    print("Loading test (out) data...")

                self.ts_out_data = USPTO_AugmentedDataset(self.data_path, "test_out_x1", self.tokenizer, self.vocabulary, verbose=False)
                self.ts_out_dataloader = DataLoader(self.ts_out_data, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, collate_fn=self._pad_collate)


    def _pad_collate(self, batch):
        """
            This function is used to create a batch of padded sequences
        """
        (xx, yy) = zip(*batch)
        x_lens = [len(x) for x in xx]
        y_lens = [len(y) for y in yy]

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=self.vocabulary["<pad>"])
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=self.vocabulary["<pad>"])

        return xx_pad, yy_pad, x_lens, y_lens

    def train_dataloader(self):
        return DataLoader(self.tr_data, batch_size=self.batch_size, shuffle=True, num_workers=self.n_workers, collate_fn=self._pad_collate)

    def val_dataloader(self):
        return DataLoader(self.vl_data, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, collate_fn=self._pad_collate)

    #def test_dataloader(self):
    #    return DataLoader(self.ts_in_data, batch_size=self.batch_size, shuffle=False, collate_fn=self._pad_collate)

    #def predict_dataloader(self):
    #    return DataLoader(self.ts_in_data, batch_size=self.batch_size, shuffle=False, collate_fn=self._pad_collate)

if __name__ == "__main__":
    try_dataset = USPTO_AugmentedDataModule(
        "./data", batch_size=128,
        type=1, n_augm=1, represent="smiles",
        verbose=True, n_workers=1,
        cross_distribution=False
    )