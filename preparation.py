import torch
from typing import List, Dict, Any
from google.colab import userdata
from transformers import EsmTokenizer

def create_esm_tokenizer(model_name: str):
  # Instantiate EsmTokenizer using from_pretrained, similar to BertTokenizerFast
  # We'll use the same model as for the EsmForSequenceClassification later for consistency.
  return EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D",
                                       token=userdata.get("Huggingface"))
tokenizer = create_esm_tokenizer()


from functools import partial
def preprocess_batch(examples, 
                     unique_items:List[str], 
                     label2id: Dict[str, int],
                     tokenizer = tokenizer, 
                     transforms=None, 
                     max_length=1024):
    """
    Preprocesses a batch of items from the dataset, and gets them ready for model training.
    Removes padding from tokenizer calls, as padding will be handled by the data_collate_function.

    Parameters:
    **examples** a dataset items (dict of lists), which contain the preprocessed samples.
    **tokenizer** an instance of the ProtBERT tokenizer, in order to perform the tokenization.
    **transforms** Any data-augmeentation transform instructions. None available, atm.
    **max_length** The maximum sequence length for tokenization. Sequences longer than this will be truncated.

    Returns: preprocessed dataset items (dict of lists), ready for passing to the model
    """
    # Common part for both batched and single example processing
    sequences_processed = [item for item in examples["sequence"]] if isinstance(examples["sequence"], list) else [examples["sequence"]]

    # Tokenize with truncation, return as Python lists of integers
    tokenized_output = tokenizer(sequences_processed, max_length=max_length, truncation=True)

    # Process labels - if they are there.
    labels_list = []
    if "labels" in examples:
        go_term_lists_processed = examples["labels"] if isinstance(examples["labels"], list) else [examples["labels"]]

        for go_terms_str in go_term_lists_processed:
            labels = torch.full((len(unique_items),), 0, dtype=torch.float)
            for item in go_terms_str.split(" "):
                if item:
                    labels[label2id[item]] = 1
            labels_list.append(labels)

    if isinstance(examples["protein"], list):
        output_dict = {
            "protein": examples["protein"],
            "input_ids": [torch.tensor(ids, dtype=torch.long) for ids in tokenized_output["input_ids"]],
            "attention_mask": [torch.tensor(ids, dtype=torch.long) for ids in tokenized_output["attention_mask"]]
        }
        # ESM models don't use token_type_ids, so we make it optional
        if "token_type_ids" in tokenized_output:
            output_dict["token_type_ids"] = [torch.tensor(ids, dtype=torch.long) for ids in tokenized_output["token_type_ids"]]

        if "labels" in examples:
            output_dict["labels"] = labels_list
        return output_dict
    else:
        output_dict = {
            "protein": examples["protein"],
            "input_ids": torch.tensor(tokenized_output["input_ids"][0], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized_output["attention_mask"][0], dtype=torch.long)
        }
        if "token_type_ids" in tokenized_output:
            output_dict["token_type_ids"] = torch.tensor(tokenized_output["token_type_ids"][0], dtype=torch.long)

        if "labels" in examples:
            output_dict["labels"] = labels_list[0]
        return output_dict

def data_collate_function(preprocessed_batch: List[Dict[str, Any]], max_length: int)-> Dict[str, Any]:
    """
    Stacks together groups of preprocessed samples into batches for our model,
    applying dynamic padding to the maximum sequence length within the batch.

    Params:
    **preprocessed_batch**: A list of Dicts, where each Dict represents a single processed sample
                            with unpadded 1D tensors for input_ids, token_type_ids, attention_mask, and labels.
    """
    collated_data = {}

    # Determine the maximum sequence length in the current batch
    max_seq_len = max_length

    # Manually pad each item in the batch
    padded_input_ids = []
    padded_token_type_ids = []
    padded_attention_mask = []
    labels = []

    for sample in preprocessed_batch:
        current_len = len(sample["input_ids"])
        padding_len = max_seq_len - current_len

        # Pad input_ids with the tokenizer's pad_token_id
        padded_input_ids.append(torch.cat([sample["input_ids"], torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long)]))
        # Pad token_type_ids with 0
        if "token_type_ids" in sample:
            padded_token_type_ids.append(torch.cat([sample["token_type_ids"], torch.full((padding_len,), 0, dtype=torch.long)]))
        # Pad attention_mask with 0
        padded_attention_mask.append(torch.cat([sample["attention_mask"], torch.full((padding_len,), 0, dtype=torch.long)]))
        if "labels" in sample:
            labels.append(sample["labels"])

    # Stack the padded tensors to form the batch
    collated_data["input_ids"] = torch.stack(padded_input_ids)
    if padded_token_type_ids != []:
        collated_data["token_type_ids"] = torch.stack(padded_token_type_ids)
    collated_data["attention_mask"] = torch.stack(padded_attention_mask)
    if labels != []:
        collated_data["labels"] = torch.stack(labels)

    return collated_data

def gen_data_collate_function(max_length: int):
    return partial(data_collate_function, max_length=max_length)