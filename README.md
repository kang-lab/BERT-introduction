# BERT-introduction

BERT is one of the most common baseline large language model trained using a masked language modeling and next sentence prediction objectives. It was created using the transformer [architecture](https://arxiv.org/abs/1706.03762) and was introduced with [this paper](https://arxiv.org/abs/1810.04805).

This model can be loaded using [huggingface](https://huggingface.co/bert-base-uncased). Normally, there will be an introduction about how to load a model on huggingface:

```
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

For every model, we always need to load a tokenizer and a model. You can swap 'bert-base-uncased' with other checkpoints such as 'bert-base-chinese' or 'bert-large-uncased' for different use cases.

Another very useful thing that we should know is the input and output of the model. BERT input size is 512 but if you use the bert-large models, then the input size is 1024. As with all other NLP models, this input (encoded_input) has to be a number tensor instead of a string (text). There are 2 interesting outputs for this model:
  - 'last_hidden_state'
  - 'pooler_output'


Other resources:
1. [BERT 101 by huggingface](https://huggingface.co/blog/bert-101)
