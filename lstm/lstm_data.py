from torch.utils.data import Dataset, DataLoader
import torch


class MoETraceDataset(Dataset):
    def __init__(self, traces, labels):
        # Traces and labels are now just standard Python lists
        self.traces = traces
        self.labels = labels

        # Auto-detect configuration
        # traces[0] is a NumPy array of shape (Seq_Len, Num_Layers, Top_K)
        first_trace_shape = self.traces[0].shape

        self.detected_num_layers = first_trace_shape[1]
        self.detected_top_k = first_trace_shape[2]

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, idx):
        # Directly convert the NumPy array to a Tensor in one step
        x_tensor = torch.tensor(self.traces[idx], dtype=torch.long)

        # Ensure labels are float32 for BCEWithLogitsLoss (assuming binary classification)
        y_label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return x_tensor, y_label


def pad_collate(batch):
    """
    Pads variable-length sequences in a batch to the max length.
    """
    # batch is a list of tuples (x_tensor, y_label)
    xx, yy = zip(*batch)

    x_lens = torch.tensor([len(x) for x in xx], dtype=torch.int64)

    # Pad the prompts with 0s so they are all the same length as the longest one
    # Output shape: (Batch, Max_Len, Num_Layers, Top_K)
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)

    # Stack labels into shape (Batch,)
    yy = torch.stack(yy)

    return xx_pad, yy, x_lens


def get_dataLoader(dataset, batch_size, shuffle, collate_fn=pad_collate):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
