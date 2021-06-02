import torch

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained("bert-base-uncased")

def encode_data(dataset, tokenizer, max_seq_length=128):
    """Featurizes the dataset into input IDs and attention masks for input into a
     transformer-style model.

     NOTE: This method should featurize the entire dataset simultaneously,
     rather than row-by-row.

  Args:
    dataset: A Pandas dataframe containing the data to be encoded.
    tokenizer: A transformers.PreTrainedTokenizerFast object that is used to
      tokenize the data.
    max_seq_length: Maximum sequence length to either pad or truncate every
      input example to.

  Returns:
    input_ids: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing token IDs for the data.
    attention_mask: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing attention masks for the data.
  """
    ## TODO: Tokenize the questions and passages using both truncation and padding.
    ## Use the tokenizer provided in the argument and see the code comments above for
    ## more details.
    tokenized = tokenizer(dataset['question'].tolist(),dataset['passage'].tolist(),truncation=True, padding='max_length',max_length=max_seq_length)
    ids = tokenized.get('input_ids')
    mask_lst = tokenized.get('attention_mask')
    ids = torch.LongTensor(ids)
    attention_mask = torch.LongTensor(mask_lst)
    return ids , attention_mask
    

def extract_labels(dataset):
    """Converts labels into numerical labels.

  Args:
    dataset: A Pandas dataframe containing the labels in the column 'label'.

  Returns:
    labels: A list of integers corresponding to the labels for each example,
      where 0 is False and 1 is True.
  """
    ## TODO: Convert the labels to a numeric format and return as a list.

    res_lst = []
    for elem in dataset['label'].tolist():
        res_lst.append( int(elem == True) )
    return res_lst 
