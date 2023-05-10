import numpy as np
import pandas as pd
import torch
import os
import wfdb
from tqdm import tqdm
import pickle
import ast

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from torch.utils.data import TensorDataset

from .base import SequenceDataset, default_data_path


class PTBXL(SequenceDataset):
    # Based on https://github.com/helme/ecg_ptbxl_benchmarking/blob/master/code/experiments/scp_experiment.py
    # and https://github.com/helme/ecg_ptbxl_benchmarking/blob/master/code/utils/utils.py
    _name_ = "ptbxl"
    # commented with dimensions for task=="all"
    d_input: int  # = 12
    d_output: int  # = 71
    l_output: int  # = 0
    L: int  # = 1000

    @property
    def init_defaults(self):
        return {
            "task": "superdiagnostic",
            "sampling_frequency": 100,
            "min_samples": 0,
            "test_fold": 10,
            "val_fold": 9,
            "train_fold": 8,
            "icbeb": False,
        }

    def prepare_data(self):
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"""
                Data directory {str(self.data_dir)} not found.
                Please download the data from https://physionet.org/content/ptb-xl/1.0.3/.
                """
            )

    def setup(self):
        assert self.task in [
            "all",
            "diagnostic",
            "subdiagnostic",
            "superdiagnostic",
            "form",
            "rhythm",
        ]

        self.data_dir = self.data_dir or default_data_path / self._name_

        # Load PTB-XL data
        data, raw_labels = self._load_dataset(
            self.data_dir, self.sampling_frequency, self.icbeb
        )

        # Preprocess label data
        labels = self._compute_label_aggregations(raw_labels, self.data_dir, self.task)

        # Select relevant data and convert to one-hot
        data, labels, Y, _ = self._select_data(
            data, labels, self.task, self.min_samples
        )
        n_classes = Y.shape[1]
        self.d_output = n_classes

        data = data.transpose(0, 2, 1)  # convert (N, L, D) to (N, D, L)
        input_shape = data[0].shape
        self.d_input, self.L = input_shape

        # 10th fold for testing (9th for now)
        X_test = data[labels.strat_fold == self.test_fold]
        y_test = Y[labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        X_val = data[labels.strat_fold == self.val_fold]
        y_val = Y[labels.strat_fold == self.val_fold]
        # rest for training
        X_train = data[labels.strat_fold <= self.train_fold]
        y_train = Y[labels.strat_fold <= self.train_fold]

        # Preprocess signal data
        X_train, X_val, X_test = self._preprocess_signals(X_train, X_val, X_test)

        self.dataset_train = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(
                y_train
            ).float(),  # convert labels to float for BCEWithLogitsLoss
        )
        self.dataset_val = TensorDataset(
            torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
        )
        self.dataset_test = TensorDataset(
            torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
        )

    def _load_dataset(self, path, sampling_rate, icbeb=False):
        if not icbeb:
            # load and convert annotation data
            Y = pd.read_csv(path / "ptbxl_database.csv", index_col="ecg_id")
            Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

            # Load raw signal data
            X = self.load_raw_data_ptbxl(Y, sampling_rate, path)
        else:
            # load and convert annotation data
            Y = pd.read_csv(path / "icbeb_database.csv", index_col="ecg_id")
            Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

            # Load raw signal data
            X = self.load_raw_data_icbeb(Y, sampling_rate, path)

        return X, Y

    def load_raw_data_ptbxl(self, df, sampling_rate, path):
        if sampling_rate == 100:
            if os.path.exists(path / "raw100.npy"):
                data = np.load(path / "raw100.npy", allow_pickle=True)
            else:
                data = [wfdb.rdsamp(path / f) for f in tqdm(df.filename_lr)]
                data = np.array([signal for signal, meta in data])
                pickle.dump(data, open(path / "raw100.npy", "wb"), protocol=4)
        elif sampling_rate == 500:
            if os.path.exists(path / "raw500.npy"):
                data = np.load(path / "raw500.npy", allow_pickle=True)
            else:
                data = [wfdb.rdsamp(path / f) for f in tqdm(df.filename_hr)]
                data = np.array([signal for signal, meta in data])
                pickle.dump(data, open(path / "raw500.npy", "wb"), protocol=4)
        return data

    def load_raw_data_icbeb(self, df, sampling_rate, path):

        if sampling_rate == 100:
            if os.path.exists(path / "raw100.npy"):
                data = np.load(path / "raw100.npy", allow_pickle=True)
            else:
                data = [
                    wfdb.rdsamp(path / "records100/" + str(f)) for f in tqdm(df.index)
                ]
                data = np.array([signal for signal, meta in data])
                pickle.dump(data, open(path / "raw100.npy", "wb"), protocol=4)
        elif sampling_rate == 500:
            if os.path.exists(path / "raw500.npy"):
                data = np.load(path / "raw500.npy", allow_pickle=True)
            else:
                data = [
                    wfdb.rdsamp(path / "records500/" + str(f)) for f in tqdm(df.index)
                ]
                data = np.array([signal for signal, meta in data])
                pickle.dump(data, open(path / "raw500.npy", "wb"), protocol=4)
        return data

    def _select_data(self, XX, YY, ctype, min_samples):
        # convert multilabel to multi-hot
        mlb = MultiLabelBinarizer()

        if ctype == "diagnostic":
            X = XX[YY.diagnostic_len > 0]
            Y = YY[YY.diagnostic_len > 0]
            mlb.fit(Y.diagnostic.values)
            y = mlb.transform(Y.diagnostic.values)
        elif ctype == "subdiagnostic":
            counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
            counts = counts[counts > min_samples]
            YY.subdiagnostic = YY.subdiagnostic.apply(
                lambda x: list(set(x).intersection(set(counts.index.values)))
            )
            YY["subdiagnostic_len"] = YY.subdiagnostic.apply(lambda x: len(x))
            X = XX[YY.subdiagnostic_len > 0]
            Y = YY[YY.subdiagnostic_len > 0]
            mlb.fit(Y.subdiagnostic.values)
            y = mlb.transform(Y.subdiagnostic.values)
        elif ctype == "superdiagnostic":
            counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
            counts = counts[counts > min_samples]
            YY.superdiagnostic = YY.superdiagnostic.apply(
                lambda x: list(set(x).intersection(set(counts.index.values)))
            )
            YY["superdiagnostic_len"] = YY.superdiagnostic.apply(lambda x: len(x))
            X = XX[YY.superdiagnostic_len > 0]
            Y = YY[YY.superdiagnostic_len > 0]
            mlb.fit(Y.superdiagnostic.values)
            y = mlb.transform(Y.superdiagnostic.values)
        elif ctype == "form":
            # filter
            counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
            counts = counts[counts > min_samples]
            YY.form = YY.form.apply(
                lambda x: list(set(x).intersection(set(counts.index.values)))
            )
            YY["form_len"] = YY.form.apply(lambda x: len(x))
            # select
            X = XX[YY.form_len > 0]
            Y = YY[YY.form_len > 0]
            mlb.fit(Y.form.values)
            y = mlb.transform(Y.form.values)
        elif ctype == "rhythm":
            # filter
            counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
            counts = counts[counts > min_samples]
            YY.rhythm = YY.rhythm.apply(
                lambda x: list(set(x).intersection(set(counts.index.values)))
            )
            YY["rhythm_len"] = YY.rhythm.apply(lambda x: len(x))
            # select
            X = XX[YY.rhythm_len > 0]
            Y = YY[YY.rhythm_len > 0]
            mlb.fit(Y.rhythm.values)
            y = mlb.transform(Y.rhythm.values)
        elif ctype == "all":
            # filter
            counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
            counts = counts[counts > min_samples]
            YY.all_scp = YY.all_scp.apply(
                lambda x: list(set(x).intersection(set(counts.index.values)))
            )
            YY["all_scp_len"] = YY.all_scp.apply(lambda x: len(x))
            # select
            X = XX[YY.all_scp_len > 0]
            Y = YY[YY.all_scp_len > 0]
            mlb.fit(Y.all_scp.values)
            y = mlb.transform(Y.all_scp.values)
        else:
            pass

        return X, Y, y, mlb

    def _preprocess_signals(self, X_train, X_validation, X_test):
        # Standardize data such that mean 0 and variance 1
        ss = StandardScaler()
        ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

        return (
            self._apply_standardizer(X_train, ss),
            self._apply_standardizer(X_validation, ss),
            self._apply_standardizer(X_test, ss),
        )

    def _apply_standardizer(self, X, ss):
        X_tmp = []
        for x in X:
            x_shape = x.shape
            X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
        X_tmp = np.array(X_tmp)
        return X_tmp

    def _compute_label_aggregations(self, df, folder, ctype):

        df["scp_codes_len"] = df.scp_codes.apply(lambda x: len(x))

        aggregation_df = pd.read_csv(folder / "scp_statements.csv", index_col=0)

        if ctype in ["diagnostic", "subdiagnostic", "superdiagnostic"]:

            def aggregate_all_diagnostic(y_dic):
                tmp = []
                for key in y_dic.keys():
                    if key in diag_agg_df.index:
                        tmp.append(key)
                return list(set(tmp))

            def aggregate_subdiagnostic(y_dic):
                tmp = []
                for key in y_dic.keys():
                    if key in diag_agg_df.index:
                        c = diag_agg_df.loc[key].diagnostic_subclass
                        if str(c) != "nan":
                            tmp.append(c)
                return list(set(tmp))

            def aggregate_diagnostic(y_dic):
                tmp = []
                for key in y_dic.keys():
                    if key in diag_agg_df.index:
                        c = diag_agg_df.loc[key].diagnostic_class
                        if str(c) != "nan":
                            tmp.append(c)
                return list(set(tmp))

            diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
            if ctype == "diagnostic":
                df["diagnostic"] = df.scp_codes.apply(aggregate_all_diagnostic)
                df["diagnostic_len"] = df.diagnostic.apply(lambda x: len(x))
            elif ctype == "subdiagnostic":
                df["subdiagnostic"] = df.scp_codes.apply(aggregate_subdiagnostic)
                df["subdiagnostic_len"] = df.subdiagnostic.apply(lambda x: len(x))
            elif ctype == "superdiagnostic":
                df["superdiagnostic"] = df.scp_codes.apply(aggregate_diagnostic)
                df["superdiagnostic_len"] = df.superdiagnostic.apply(lambda x: len(x))
        elif ctype == "form":
            form_agg_df = aggregation_df[aggregation_df.form == 1.0]

            def aggregate_form(y_dic):
                tmp = []
                for key in y_dic.keys():
                    if key in form_agg_df.index:
                        c = key
                        if str(c) != "nan":
                            tmp.append(c)
                return list(set(tmp))

            df["form"] = df.scp_codes.apply(aggregate_form)
            df["form_len"] = df.form.apply(lambda x: len(x))
        elif ctype == "rhythm":
            rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

            def aggregate_rhythm(y_dic):
                tmp = []
                for key in y_dic.keys():
                    if key in rhythm_agg_df.index:
                        c = key
                        if str(c) != "nan":
                            tmp.append(c)
                return list(set(tmp))

            df["rhythm"] = df.scp_codes.apply(aggregate_rhythm)
            df["rhythm_len"] = df.rhythm.apply(lambda x: len(x))
        elif ctype == "all":
            df["all_scp"] = df.scp_codes.apply(lambda x: list(set(x.keys())))

        return df
