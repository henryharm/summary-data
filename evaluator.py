import random

from rouge_score import rouge_scorer
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)


def print_samples(out, trg):
    for el in range(5):
        index = random.randint(0, len(trg) - 1)
        logging.debug("Sample target: \n\n{}\n\n".format(trg[index]))
        logging.debug("Sample output: \n\n{}\n\n".format(out[index]))


class Evaluator:
    def __init__(self, tokenizer):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.tokenizer = tokenizer

    def averaged_rouge_score(self, out, trg, batched=True):
        avg_score = {'rouge1': {'fmeasure': 0.0, 'precision': 0.0, 'recall': 0.0},
                     'rouge2': {'fmeasure': 0.0, 'precision': 0.0, 'recall': 0.0},
                     'rougeL': {'fmeasure': 0.0, 'precision': 0.0, 'recall': 0.0}}

        #logging.debug("\nOut: {} \n".format(np.array(out).shape))
        #logging.debug("\nTrg: {} \n".format(np.array(trg).shape))
        #logging.debug("\nTarget sample: {} \n".format(trg[0]))

        if batched:
            for i, batch in enumerate(out):
                for j, row in enumerate(batch):
                    score = self.scorer.score(row, trg[i][j])
                    for rouge in avg_score:
                        avg_score[rouge]['fmeasure'] += score[rouge].fmeasure
                        avg_score[rouge]['precision'] += score[rouge].precision
                        avg_score[rouge]['recall'] += score[rouge].recall
        else:
            for i, row in enumerate(out):
                score = self.scorer.score(row, trg[i])
                for rouge in avg_score:
                    avg_score[rouge]['fmeasure'] += score[rouge].fmeasure
                    avg_score[rouge]['precision'] += score[rouge].precision
                    avg_score[rouge]['recall'] += score[rouge].recall


        length = len(out)
        for rouge in avg_score:
            for measure in avg_score[rouge]:
                avg_score[rouge][measure] = avg_score[rouge][measure] / length

        return avg_score

    def compute_metrics(self, pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        logging.debug("\nOut: {} \n".format(np.array(pred_str).shape))
        logging.debug("\nTrg: {} \n".format(np.array(label_str).shape))

        rouge_output = self.averaged_rouge_score(pred_str, label_str, False)

        return rouge_output

    def evaluate_model(self, test_loader, model):
        result = {"transcripts": [], "pred": []}

        for i, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")

            result["transcripts"].append(self.tokenizer.batch_decode(input_ids, skip_special_tokens=True))

            outputs = model.generate(input_ids, attention_mask=attention_mask)

            # all special tokens including will be removed
            output_str = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            result["pred"].append(output_str)

        pred_str = np.array(result["pred"])
        label_str = np.array(result["transcripts"])
        #logging.debug("averaging score row {} and trg {}".format(pred_str, label_str))

        rouge_output = self.averaged_rouge_score(pred_str, label_str)

        return rouge_output, pred_str, label_str
