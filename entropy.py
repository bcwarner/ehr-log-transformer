# Evaluates the calculated entropy values for the test set.
import argparse
import inspect
import os
import sys
from collections import defaultdict

import scipy.stats
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from tabulate import tabulate
from matplotlib import pyplot as plt

from model.model import EHRAuditGPT2
from model.modules import EHRAuditPretraining, EHRAuditDataModule
from model.data import timestamp_space_calculation
from model.vocab import EHRVocab
import tikzplotlib
import numpy as np

METRIC_NAME_COL = 0
PAT_ID_COL = 1
ACCESS_TIME_COL = 2


class Experiment:
    def __init__(
        self,
        config: dict,
        path_prefix: str,
        vocab: EHRVocab,
        dm: EHRAuditDataModule,
        *args,
        **kwargs,
    ):
        self.config = config
        self.path_prefix = path_prefix
        self.vocab = vocab
        self.dm = dm

    def on_row(
        self,
        row=None,
        prev_row=None,
        row_loss=None,
        prev_row_loss=None,
        batch_no=None,
        batch_idx=None,
    ):
        pass

    def on_finish(self):
        pass

    def on_batch(self, sequence):
        """
        Returns True if the batch should be processed.
        """
        return True

    def samples_seen(self):
        return -1


class EntropySwitchesExperiment(Experiment):
    def __init__(
        self,
        config: dict,
        path_prefix: str,
        vocab: EHRVocab,
        *args,
    ):
        """
        Measures the entropy of the nth switch during a session vs. the entropy.
        Also compares the entropy of all switches during a session vs. non-switches.
        """
        super().__init__(config, path_prefix, vocab)

        self.switch_entropies_before: defaultdict[int, list] = defaultdict(
            list
        )  # Switch no => list of entropies
        self.switch_entropies_after: defaultdict[int, list] = defaultdict(
            list
        )  # Switch no => list of entropies
        self.non_switch_entropies = []  # List of entropies for non-switches
        self.batch_ct = defaultdict(int)  # Batch no => current row
        self._samples_seen = 0

    def on_row(
        self,
        row=None,
        row_loss=None,
        prev_row=None,
        prev_row_loss=None,
        batch_no=None,
        batch_idx=None,
    ):
        if prev_row is None:
            return
        if batch_no not in self.batch_ct:
            self._samples_seen += 1
        self.batch_ct[batch_no] += 1
        if prev_row[PAT_ID_COL] != row[PAT_ID_COL]:
            switch_ct = self.batch_ct[batch_no]
            self.switch_entropies_before[switch_ct].append(prev_row_loss)
            self.switch_entropies_after[switch_ct].append(row_loss)
        else:
            self.non_switch_entropies.append(row_loss)

    def on_batch(self, sequence):
        return True

    def on_finish(self):
        # Plot the entropy of the nth switch during a session vs. the entropy.
        # Also compare the entropy of all switches during a session vs. non-switches.
        switch_entropies_before_mean = []
        switch_entropies_before_std = []
        switch_entropies_after_mean = []
        switch_entropies_after_std = []
        for switch_ct in self.switch_entropies_before:
            switch_entropies_before_mean.append(
                np.mean(self.switch_entropies_before[switch_ct])
            )
            switch_entropies_after_std.append(
                np.std(self.switch_entropies_after[switch_ct])
            )
            switch_entropies_after_mean.append(
                np.mean(self.switch_entropies_after[switch_ct])
            )
            switch_entropies_before_std.append(
                np.std(self.switch_entropies_before[switch_ct])
            )

        # Plot the entropy of the nth switch during a session vs. the entropy.
        x = np.arange(1, len(switch_entropies_before_mean) + 1)
        max_n = 10
        plt.errorbar(
            x[:max_n],
            switch_entropies_before_mean[:max_n],
            yerr=switch_entropies_before_std[:max_n],
            label="Before",
        )
        plt.errorbar(
            x[:max_n],
            switch_entropies_after_mean[:max_n],
            yerr=switch_entropies_after_std[:max_n],
            label="After",
        )
        plt.xlabel("Switch")
        plt.ylabel("Entropy")
        plt.legend()
        res_path = os.path.normpath(
            os.path.join(self.path_prefix, self.config["results_path"])
        )
        plt.savefig(os.path.normpath(os.path.join(res_path, "entropy_switches.png")))

        switch_entropies_before_all = []
        switch_entropies_after_all = []
        for switch_ct in self.switch_entropies_before:
            switch_entropies_before_all.extend(self.switch_entropies_before[switch_ct])
            switch_entropies_after_all.extend(self.switch_entropies_after[switch_ct])

        # Also compare the entropy of all switches during a session vs. non-switches.
        plt.clf()
        plt.boxplot(
            [
                self.non_switch_entropies,
                switch_entropies_before_all,
                switch_entropies_after_all,
            ]
        )
        plt.xticks([1, 2, 3], ["Non-switch", "Before", "After"])
        plt.ylabel("Entropy")
        plt.savefig(
            os.path.normpath(
                os.path.join(res_path, "entropy_switches_vs_non_switches.png")
            )
        )

        # Make a probability distribution funciton of the entropy of switches vs. non-switches using matplotlib
        plt.clf()
        plt.hist(
            [
                self.non_switch_entropies,
                switch_entropies_before_all,
                switch_entropies_after_all,
            ],
            bins=50,
            density=True,
            histtype="step",
            label=["Non-switch", "Before", "After"],
        )
        plt.legend()
        plt.xlabel("Entropy")
        plt.ylabel("Probability")
        plt.savefig(
            os.path.normpath(
                os.path.join(res_path, "entropy_switches_vs_non_switches_cdf.png")
            )
        )

        # Plot the log entropy of switches vs. non-switches using matplotlib
        plt.clf()
        plt.hist(
            [
                np.log(self.non_switch_entropies),
                np.log(switch_entropies_before_all),
                np.log(switch_entropies_after_all),
            ],
            bins=50,
            density=True,
            histtype="step",
            label=["Non-switch", "Before", "After"],
        )
        plt.legend()
        plt.xlabel("Log entropy")
        plt.ylabel("Probability")
        plt.savefig(
            os.path.normpath(
                os.path.join(res_path, "log_entropy_switches_vs_non_switches_cdf.png")
            )
        )

        # Calculate the p-value that the distributions are the same using scipy
        _, p_before = scipy.stats.ttest_ind(
            self.non_switch_entropies, switch_entropies_before_all
        )
        _, p_after = scipy.stats.ttest_ind(
            self.non_switch_entropies, switch_entropies_after_all
        )
        print(f"Before t-test p-value: {p_before}")
        print(f"After t-test p-value: {p_after}")

    def samples_seen(self):
        return self._samples_seen


class SecureChatEntropy(Experiment):
    def __init__(self, config, path_prefix, vocab):
        super().__init__(config, path_prefix, vocab)
        # Find the vocab elements that have "secure chat" in them
        self.secure_chat_vocab = []
        self.secure_chat_vocab_names = []
        for vk, vv in self.vocab.field_tokens["METRIC_NAME"].items():
            if "secure chat" in vk.lower():
                self.secure_chat_vocab.append(vv)
                self.secure_chat_vocab_names.append(vk)

        self.entropy_by_type = defaultdict(
            list
        )  # Secure chat type => list of entropies
        # Are they higher overall for when seen in a secure chat?
        self.entropy_present = list()  # List of entropies seen in sequences visited
        self._samples_seen = 0

    def on_batch(self, sequence):
        # Only examine sequences with secure chat
        res = any([x in sequence for x in self.secure_chat_vocab])
        self._samples_seen += int(res)
        return res

    def on_row(
        self,
        row=None,
        prev_row=None,
        row_loss=None,
        prev_row_loss=None,
        batch_no=None,
        batch_idx=None,
    ):
        # Get the METRIC_NAME token for this row.
        metric_name_token = row[METRIC_NAME_COL]
        if metric_name_token in self.secure_chat_vocab:
            self.entropy_by_type[metric_name_token].append(row_loss)
            self._samples_seen += 1
        else:
            self.entropy_present.append(row_loss)

    def on_finish(self):
        # Plot the entropy of each type of secure chat as a histogram.
        plt.clf()
        for token, name in zip(self.secure_chat_vocab, self.secure_chat_vocab_names):
            plt.hist(
                self.entropy_by_type[token],
                bins=50,
                density=True,
                histtype="step",
                label=name,
            )
        # Also plot the entropy of all tokens seen in secure chat as a histogram.
        plt.hist(
            self.entropy_present,
            bins=50,
            density=True,
            histtype="step",
            label="Other Session Actions",
        )
        plt.legend()
        plt.xlabel("Entropy")
        plt.ylabel("Probability")
        res_path = os.path.normpath(
            os.path.join(self.path_prefix, self.config["results_path"])
        )
        plt.savefig(
            os.path.normpath(os.path.join(res_path, "entropy_secure_chat_types.png"))
        )

        # Plot the entropy of each type of secure chat as a boxplot.
        plt.clf()
        plt.boxplot(
            [self.entropy_by_type[token] for token in self.secure_chat_vocab]
            + [self.entropy_present]
        )
        plt.xticks(
            range(1, len(self.secure_chat_vocab) + 2),
            self.secure_chat_vocab_names + ["Other Session Actions"],
            rotation=90,
        )
        plt.ylabel("Entropy")
        plt.savefig(
            os.path.normpath(
                os.path.join(res_path, "entropy_secure_chat_types_boxplot.png")
            )
        )

    def samples_seen(self):
        return self._samples_seen


class PatientsSessionsEntropyExperiment(Experiment):
    def __init__(self, config, path_prefix, vocab):
        super().__init__(config, path_prefix, vocab)
        # Get the entropy of each session as a function of the number of patients
        self.entropy_by_patient_count_mean: defaultdict[int, list] = defaultdict(
            list
        )  # Number of patients => list of entropies
        self.entropy_by_patient_count_std: defaultdict[int, list] = defaultdict(
            list
        )  # Number of patients => list of entropies

        # Iteration variables
        self.cur_batch = -1
        self.seen_patients = set()
        self.entropies = list()
        self._samples_seen = 0

    def on_row(
        self,
        row=None,
        prev_row=None,
        row_loss=None,
        prev_row_loss=None,
        batch_no=None,
        batch_idx=None,
    ):
        # Get the patient ID
        patient_id = row[PAT_ID_COL]
        # If we've seen a new patient, record the entropy of the previous patient
        if self.cur_batch != batch_no and self.cur_batch != -1:
            self.entropy_by_patient_count_mean[len(self.seen_patients)].append(
                np.mean(self.entropies)
            )
            self.entropy_by_patient_count_std[len(self.seen_patients)].append(
                np.std(self.entropies)
            )
            self.seen_patients = set()
            self.entropies = list()
            self._samples_seen += 1
        self.cur_batch = batch_no

        if patient_id not in self.seen_patients:
            self.seen_patients.add(patient_id)
            # Record the entropy of this row
        self.entropies.append(row_loss)

    def on_finish(self):
        # Scatter plot of mean entropy by number of patients
        plt.clf()
        points = []
        for k, v in self.entropy_by_patient_count_mean.items():
            for y in v:
                points.append((k, y))
        x, y = zip(*points)
        # Make x the log scale
        plt.scatter(x, y)
        plt.xlabel("Number of Patients")
        plt.ylabel("Mean Entropy")
        plt.gca().set_xscale("log")
        # Trendline w/ correlation
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        r = np.corrcoef(x, y)[0, 1]
        x_ = np.linspace(
            min(self.entropy_by_patient_count_mean.keys()),
            max(self.entropy_by_patient_count_mean.keys()),
            100,
        )
        plt.plot(x_, p(x_), "r--", label="Trendline (r={:.2f})".format(r))
        # Trendline for patient counts above 10 patients
        x_filt, y_filt = zip(*[(x, y) for x, y in points if x > 10])
        z = np.polyfit(x_filt, y_filt, 1)
        p = np.poly1d(z)
        r = np.corrcoef(x_filt, y_filt)[0, 1]
        x_ = np.linspace(10, max(x_), 100)
        plt.plot(x_, p(x_), "g--", label="10+ Patients (r={:.2f})".format(r))
        plt.title("Mean Entropy by Number of Patients in a Session")
        plt.legend()
        plt.savefig(
            os.path.normpath(
                os.path.join(
                    self.path_prefix,
                    self.config["results_path"],
                    "entropy_by_patient_count.png",
                )
            )
        )

    def samples_seen(self):
        return self._samples_seen


class TimeEntropyExperiment(Experiment):
    """
    Measures the entropy as a function of the time delta.
    """

    def __init__(self, config, path_prefix, vocab):
        super().__init__(config, path_prefix, vocab)
        self.entropies_by_time_delta = defaultdict(list)
        self.time_delta_count = defaultdict(int)
        self._samples_seen = 0

    def on_batch(self, sequence):
        self._samples_seen += 1
        return True

    def on_row(
        self,
        row=None,
        prev_row=None,
        row_loss=None,
        prev_row_loss=None,
        batch_no=None,
        batch_idx=None,
    ):
        # Get the time delta
        time_delta = row[ACCESS_TIME_COL]
        # Record the entropy of this row
        self.entropies_by_time_delta[time_delta].append(row_loss)
        self.time_delta_count[time_delta] += 1

    def samples_seen(self):
        return self._samples_seen

    def on_finish(self):
        # Plot the entropy by time delta as a combined barchart showing the % frequency of each time delta
        # as well as their average entropy w/ error bars
        plt.clf()
        plt.figure(figsize=(20, 10))
        # Get the average entropy for each time delta
        time_delta_to_mean_entropy = {
            k: np.mean(v) for k, v in self.entropies_by_time_delta.items()
        }
        time_delta_err = {
            k: np.std(v) / np.sqrt(self.time_delta_count[k])
            for k, v in self.entropies_by_time_delta.items()
        }
        # Get the frequency of each time delta
        n = sum(self.time_delta_count.values())
        time_delta_to_freq = {k: v / n for k, v in self.time_delta_count.items()}
        time_deltas = sorted(self.time_delta_count.keys())
        # Combined plot
        fig, ax1 = plt.subplots()
        # Get the timestamp bins.
        timestamps = timestamp_space_calculation(
            list(config["timestamp_bins"].values())
        )
        # Format as token (timestamp)
        ax_labels = ["{} ({})".format(x, y) for x, y in zip(timestamps, time_deltas)]
        # Plot the frequency of each time delta with labels above the bars
        ax1.bar(
            time_deltas,
            [time_delta_to_freq[x] for x in time_deltas],
            label="Frequency",
        )
        ax1.set_xticks(range(len(time_deltas)), labels=ax_labels)
        ax1.set_ylim(0, 1.1)
        # Print a frequency label above each bar
        for k, v in time_delta_to_freq.items():
            ax1.text(
                k,
                v,
                "{:.2f}".format(v),
                color="black",
                ha="center",
            )
        ax2 = ax1.twinx()
        # Plot the average entropy for each time delta
        ax2.errorbar(
            time_deltas,
            [time_delta_to_mean_entropy[x] for x in time_deltas],
            yerr=[time_delta_err[x] for x in time_deltas],
            label="Mean Entropy",
            color="orange",
        )
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.xlabel("Time Delta (seconds)")
        plt.xticks(
            time_deltas,
            [str(x) for x in time_deltas],
            rotation=90,
        )
        ax1.set_ylabel("Frequency")
        ax2.set_ylabel("Mean Entropy")
        plt.title("Frequency and Mean Entropy by Time Delta")
        plt.savefig(
            os.path.normpath(
                os.path.join(
                    self.path_prefix,
                    self.config["results_path"],
                    "entropy_by_time_delta.png",
                )
            )
        )
        tikzplotlib.save(
            os.path.normpath(
                os.path.join(
                    self.path_prefix,
                    self.config["results_path"],
                    "entropy_by_time_delta.tex",
                )
            )
        )


class DayLevelPatientEntropyExperiment(Experiment):
    """
    Measures the entropy as a function of the number of unique patients seen in a day.
    """

    def __init__(self):
        super().__init__()
        self._samples_seen = 0
        self.entropies_by_batch_idx = defaultdict(list)

    def on_batch(self, sequence):
        self._samples_seen += 1
        return True

    def on_row(
        self,
        row=None,
        prev_row=None,
        row_loss=None,
        prev_row_loss=None,
        batch_no=None,
        batch_idx=None,
    ):
        self.entropies_by_batch_idx[batch_idx].append(row_loss)

    def on_finish(self):
        # Map the batch indices to the provider => day on which the batch occurred
        day_entropies = defaultdict(lambda: defaultdict(list))
        for batch_idx, entropies in self.entropies_by_batch_idx.items():
            # Get the day on which this batch occurred
            day = None
            provider = None
            # Append the entropy to day_entropies
            day_entropies[provider][day].extend(entropies)

        # Now get the number of patients per day
        # Divide the average entropy by the number of patients
        # Plot the average entropy as a function of the number of patients


if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=int, default=None, help="Model to use for pretraining."
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Maximum batches to use."
    )
    parser.add_argument(
        "--exp_suffix",
        type=str,
        default="",
        help="Suffix to add to the output file name.",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="Experiment",
        help="Experiment to run.",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Run with the validation dataset instead of the test.",
    )
    args = parser.parse_args()
    # Get the list of models from the config file
    config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "config.yaml")
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    path_prefix = ""
    for prefix in config["path_prefix"]:
        if os.path.exists(prefix):
            path_prefix = prefix
            break

    if path_prefix == "":
        raise RuntimeError("No valid drive mounted.")

    model_paths = os.path.normpath(
        os.path.join(path_prefix, config["pretrained_model_path"])
    )
    # Get recursive list of subdirectories
    model_list = []
    for root, dirs, files in os.walk(model_paths):
        # If there's a .bin file, it's a model
        if any([file.endswith(".bin") for file in files]):
            # Append the last three directories to the model list
            model_list.append(os.path.join(*root.split(os.sep)[-3:]))

    if len(model_list) == 0:
        raise ValueError(f"No models found in {format(model_paths)}")

    if args.model is None:
        print("Select a model to evaluate:")
        for i, model in enumerate(sorted(model_list)):
            print(f"{i}: {model}")

        model_idx = int(input("Model index >>>"))
    else:
        model_idx = args.model

    model_name = model_list[model_idx]
    model_path = os.path.normpath(
        os.path.join(path_prefix, config["pretrained_model_path"], model_name)
    )

    # Get the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    vocab = EHRVocab(
        vocab_path=os.path.normpath(os.path.join(path_prefix, config["vocab_path"]))
    )
    model = EHRAuditGPT2.from_pretrained(model_path, vocab=vocab)
    model.to(device)

    dm = EHRAuditDataModule(
        yaml_config_path=config_path,
        vocab=vocab,
        batch_size=1,  # Just one sample at a time
    )
    dm.setup()
    if args.val:
        dl = dm.val_dataloader()
    else:
        dl = dm.test_dataloader()

    if args.max_samples is None:
        print(f"Maximum number of samples to evaluate (0 for all, max = {len(dl)}):")
        max_samples = int(input(">>>"))
    else:
        max_samples = args.max_samples

    if max_samples == 0:
        max_samples = len(dl)
    else:
        max_samples = min(max_samples, len(dl))

    window_size = 30  # 30 action window

    # Initialize the experiments
    if "," in args.exp:
        experiments = [
            eval(exp)(config, path_prefix, vocab) for exp in args.exp.split(",")
        ]
    elif "all" in args.exp:
        # Get a list of all classes that sublcass Experiment in this file.
        exp_classes = [
            obj
            for name, obj in inspect.getmembers(sys.modules[__name__])
            if inspect.isclass(obj)
            and issubclass(obj, Experiment)
            and obj != Experiment
        ]
        experiments = [exp(config, path_prefix, vocab) for exp in exp_classes]
    else:
        experiments = [eval(args.exp)(config, path_prefix, vocab)]

    print(f"Running experiments:")
    for exp in experiments:
        print("-", type(exp).__name__)

    # Initialize progress bars for each experiment
    exp_pbar = [
        tqdm(total=max_samples, position=x, desc=type(exp).__name__)
        for x, exp in enumerate(experiments)
    ]

    # Calculate the entropy values for the test set
    ce_values = []
    batches_seen = 0
    batches_skipped = 0
    pbar = tqdm(total=max_samples, position=len(experiments), desc="Batches Seen")
    for batch_idx, batch in enumerate(dl):
        input_ids, labels = batch
        # Sliding window over the sequence
        with torch.no_grad():
            # Find the eos index
            nonzeros = (labels.view(-1) == -100).nonzero(as_tuple=True)
            if len(nonzeros[0]) == 0:
                eos_index = len(labels.view(-1)) - 1
            else:
                eos_index = nonzeros[0][0].item() - 1

            # Copy the labels and targets
            input_ids_c = torch.zeros_like(input_ids)
            labels_c = labels.clone()
            # Set the labels to -100, zero out the input_ids
            labels_c[:, :] = -100

            ce_current = []

            row_len = len(vocab.field_ids) - 1  # Exclude special fields
            row_count = (eos_index - 1) // row_len
            if row_count <= 1:  # Not applicable
                continue

            if len(experiments) > 0:
                should_on_batch = [
                    experiments[i].on_batch(input_ids[0])
                    and experiments[i].samples_seen() < max_samples
                    for i in range(len(experiments))
                ]

            if len(experiments) > 0 and not any(should_on_batch):
                if all([exp.samples_seen() >= max_samples for exp in experiments]):
                    break
                pbar.set_postfix({"Skipped": batches_skipped})
                batches_skipped += 1
                continue

            prev_row = None
            prev_row_loss = None
            # NOTE: Next-token generation != next-row generation
            # This means that we include the next two tokens in the input to avoid EOS predictions.
            for i in range(0, row_count):
                input_ids_start = i * row_len
                input_ids_end = input_ids_start + row_len
                input_ids_end_extra = input_ids_end + row_len
                # Get the current row
                input_ids_c[:, input_ids_start:input_ids_end_extra] = input_ids[
                    :, input_ids_start:input_ids_end_extra
                ]
                # Labels are next row.
                labels_row_start = (i + 1) * row_len
                labels_row_end = labels_row_start + row_len
                labels_c[:, labels_row_start:labels_row_end] = labels[
                    :, labels_row_start:labels_row_end
                ]
                if i > 0:
                    labels_c[
                        :, input_ids_start:input_ids_end
                    ] = -100  # Eliminate previous row.

                # if i >= window_size:
                #    old_row_start = (i - window_size) * row_len
                #    old_row_end = old_row_start + row_len
                #    input_ids_c[:, old_row_start:old_row_end] = 0

                # Calculate the cross entropy
                loss, _, _ = model(input_ids_c.to(device), labels=labels_c.to(device))
                # Divide the cross-entropy by the number of tokens in the row to get avg. token CE
                avg_loss = loss.item() / row_len
                ce_current.append(avg_loss)
                for j in range(len(experiments)):
                    if should_on_batch[j]:
                        # row = input_ids[:, input_ids_start:input_ids_end].tolist()[0]
                        labels_row = labels[
                            :, labels_row_start:labels_row_end
                        ].tolist()[0]
                        experiments[j].on_row(
                            row=labels_row,
                            row_loss=avg_loss,
                            prev_row=prev_row,
                            prev_row_loss=prev_row_loss,
                            batch_no=batches_seen,  # Batch count seen
                            batch_idx=batch_idx,  # Actual index of batch in dataloader
                        )
                        exp_pbar[j].n = experiments[j].samples_seen()
                        exp_pbar[j].refresh()
                prev_row = labels_row
                prev_row_loss = avg_loss

            pbar.update(1)
            ce_values.append(np.mean(ce_current))

        batches_seen += 1

        if max_samples != 0 and all(
            [exp.samples_seen() >= max_samples for exp in experiments]
        ):
            break

    for e in experiments:
        e.on_finish()

    # Print statistics about the entropy values
    stats = {
        "Mean CE": np.mean(ce_values),
        "Median CE": np.median(ce_values),
        "Max CE": np.max(ce_values),
        "Min CE": np.min(ce_values),
        "Std CE": np.std(ce_values),
        "Perplexity": np.mean(np.exp(ce_values)),
    }

    print(tabulate(stats.items(), headers=["Metric", "Value"]))

    plt.clf()
    # Plot the entropy values
    print(f"Plotting entropy values for {len(ce_values)} samples...")
    plt.hist(ce_values, bins=100)
    plt.title(f"Avg. Entropy/Token for Test Set (N = {len(ce_values)})")
    plt.xlabel("Entropy")
    plt.ylabel("Count")
    plt.savefig(
        os.path.normpath(
            os.path.join(
                path_prefix,
                config["results_path"],
                f"entropy_{len(ce_values)}_{args.exp_suffix}.png",
            )
        )
    )
    tikzplotlib.save(
        os.path.normpath(
            os.path.join(
                path_prefix,
                config["results_path"],
                f"entropy_{len(ce_values)}_{args.exp_suffix}.tex",
            )
        )
    )

    # Plot as perplexity
    print(f"Plotting perplexity values for {len(ce_values)} samples...")
    plt.clf()
    plt.hist(np.exp(ce_values), bins=100)
    plt.title(f"Avg. Perplexity/Token Values for Test Set (N = {len(ce_values)})")
    plt.xlabel("Perplexity")
    plt.ylabel("Count")
    plt.savefig(
        os.path.normpath(
            os.path.join(
                path_prefix,
                config["results_path"],
                f"perplexity_{len(ce_values)}_{args.exp_suffix}.png",
            )
        )
    )
    tikzplotlib.save(
        os.path.normpath(
            os.path.join(
                path_prefix,
                config["results_path"],
                f"perplexity_{len(ce_values)}_{args.exp_suffix}.tex",
            )
        )
    )
