import torch
from torch import nn
from typing import Dict
import pandas as pd

def process_result(logits, 
                   id2label: Dict[int, str], 
                   threshold:float=0.6, 
                   top_n: int=10):
  """
  This function will take in a single protein logit prediction and turns it into alist of predicted GO-terms
  Args:
  **logits** a single Logit array
  **threshold** A float value between 0 and 1. Only GO-terms with a score higher than this value will be considered.
  **top_n** An integer value. If no GO-terms are above the threshold, the top_n GO-terms will be returned.
  Ensure, that the length is equal
  """
  all_results = []
  # Convert logits to probabilities using sigmoid
  probabilities = torch.sigmoid(logits).squeeze()

  # Get indices of GO terms above threshold
  above_threshold_indices = torch.where(probabilities > threshold)[0]

  # If no terms above threshold, take the top_n terms
  if len(above_threshold_indices) == 0:
    top_n_values, top_n_indices = torch.topk(probabilities, k=min(top_n, len(probabilities)))
    selected_indices = top_n_indices
    selected_probabilities = top_n_values
  else:
    selected_indices = above_threshold_indices
    selected_probabilities = probabilities[above_threshold_indices]

  # Sort selected terms by probability in descending order
  sorted_probabilities, sort_indices = torch.sort(selected_probabilities, descending=True)
  sorted_indices = selected_indices[sort_indices]

  for idx, prob in zip(sorted_indices, sorted_probabilities):
    go_term = id2label[idx.item()]
    score = prob.item()
    all_results.append([go_term, f"{score:.3f}"]) # Format score to 3 decimal places

  results_df = pd.DataFrame(all_results)
  # The request specifies 3 columns, all string, no headers, no index
  return results_df

def predict_on_input(input: str,
                     threshold:float,
                     model: nn.Module,
                     device:str,
                     tokenizer):
    model.eval()
    with torch.inference_mode():
        tokenized = tokenizer(input,
                                return_tensors="pt").to(device)
        output_logits = model(**tokenized)

    output_table = process_result(output_logits.logits,
                                threshold=threshold)

    return output_table