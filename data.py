import torch
from torch.utils.data import Dataset
from torchtext.data import TabularDataset, Field


def load_data(tokenize_text, tokenize_summary, path):
    ### initlize Field for source article and target summary
    ARTICLE = Field(use_vocab=False,
                    sequential=True,
                    tokenize=tokenize_text,
                    )

    SUMMARY = Field(use_vocab=False,
                    sequential=True,
                    tokenize=tokenize_summary,
                    )

    # load train, test, val data
    train_data, valid_data, test_data = TabularDataset.splits(
        path=path,
        train='train.csv',
        validation='val.csv',
        test='test.csv',
        format='csv',
        fields=[("transcript", ARTICLE), ('summary', SUMMARY)]
    )

    return test_data, valid_data, test_data


class ERRDataset(Dataset):
    def __init__(self, torchtext_data, tokenizer):
        self.dataset = torchtext_data.examples
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        summary_idx = self.dataset[idx].summary.data
        transcript_idx = self.dataset[idx].transcript.data

        item = {key: torch.tensor(summary_idx[key]) for key in summary_idx}
        item['decoder_input_ids'] = torch.tensor(transcript_idx['input_ids'])
        item['decoder_attention_mask'] = torch.tensor(transcript_idx['attention_mask'])
        item['labels'] = torch.tensor(transcript_idx['input_ids'].copy())

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        item["labels"] = torch.tensor([-100 if token == self.tokenizer.pad_token_id else token for token in item["labels"]])

        return item

    def __len__(self):
        return len(self.dataset)
