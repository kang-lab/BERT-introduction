# BERT basics

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

For every model, we always need to load a tokenizer and a model. You can swap `'bert-base-uncased'` with other checkpoints such as `'bert-base-chinese'` or `'bert-large-uncased'` for different use cases.

Another very useful thing that we should know is the input and output of the model. BERT input size is 512 but if you use the bert-large models, then the input size is 1024. As with all other NLP models, this input (`encoded_input`) has to be a number tensor instead of a string (`text`). There are 2 interesting outputs for this model:
  - `'last_hidden_state'` has a dimension of (batch x sequence dimension x encoding dimension). This contains the final embeddings of all tokens in the sentence and can be understood as the final representation of each word in a sentence. We can apply permutation methods, such as max, mean or sum, to aggregate the embeddings into a single sentence representation.
  - `'pooler_output'` has a dimension of (batch x encoding dimension). This is the embedding of the `CLS` special token and is most commonly considered as a valid representation of the complete sentence. In fact the [BertForSequenceClassification wrapper](https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/bert/modeling_bert.py#L1576) by huggingface use pooler_output to classify a sequence.

# BERT next steps
After finding a good BERT model and checkpoints, your next step would be (1) defining a classifer and (2) train the model

## BERT classifier
To create a BERT classifier, you can either choose a huggingface wrapper or create your own task-specific wrapper.

There are many wrapper created by huggingface for a lot of use cases such as `BertForNextSentencePrediction`, `BertForSequenceClassification`, `BertForMultipleChoice`, and `BertFotQuestionAnswering`. You can check this huggingface docs for details on how to use a wrapper. They normally give an example with good explanation:
    
```
import torch
from transformers import AutoTokenizer, BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity", num_labels=num_labels)

labels = torch.tensor([1])
loss = model(**inputs, labels=labels).loss
round(loss.item(), 2)
```
  - Another approach is to make your own classifier function. I have 3 examples here ranging from very simple to more complex. The first example is a very simple classifier where I only pass the CLS token through a Linear layer after dropping 20% of the weights. The second example is from a multitask leaning experiment where I will swap out the last linear layer depends on the task. If the model was learning Task1, it would use the 1st linear layer and if it was learning Task2, the 2nd linear layer. This approach is very helpful when you want to make a model learn multiple tasks at the same time to increase external validity or generalizability. The last example is an experiment where I used 2 seperate learning models to read 2 different inputs and output a single decision probability after combining the output of the 2 models. Basically, you will have more flexibility when designing experiments with custom functions.

```
class severity_classifier(nn.Module):
    def __init__(self, pretrained_LM):
        super(severity_classifier, self).__init__()

        self.bert = pretrained_LM 
        self.dropout = nn.Dropout(0.2)
        self.dense1 = nn.Linear(768,1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask).last_hidden_state[:, 0, :]
        x = self.dense1(self.dropout(cls_hs))
        
        return x
```
```
class severity_classifier_multitask(nn.Module):
    def __init__(self, pretrained_LM):
        super(severity_classifier_multitask, self).__init__()

        self.bert = pretrained_LM 
        self.dropout = nn.Dropout(0.2)
        self.dense1 = nn.Linear(768,1)
        self.dense2 = nn.Linear(768,1)

    def forward(self, sent_id, mask, task):
        cls_hs = self.bert(sent_id, attention_mask=mask).last_hidden_state[:, 0, :]
        
        if task == 'Task1':
            x = self.dense1(self.dropout(cls_hs))
        else:
            x = self.dense2(self.dropout(cls_hs))
            
        return x
```
```
class vit2x(nn.Module):
    def __init__(self, model_name, hidden_dropout_prob, attention_probs_dropout_prob, attention_heads, hidden_layers):
        super(vit2x, self).__init__()

        self.vit = ViTModel.from_pretrained(model_name,
                                            hidden_dropout_prob=hidden_dropout_prob,
                                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                                            num_hidden_layers=hidden_layers,
                                            num_attention_heads=attention_heads)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(768, 1)
        )

    def forward(self, cc, mlo):
        cc_output = self.vit(cc)
        mlo_output = self.vit(mlo)

        cc_output = cc_output.last_hidden_state
        mlo_output = mlo_output.last_hidden_state

        cc_output = cc_output[:, 0, :]
        mlo_output = mlo_output[:, 0, :]

        viewpool = torch.max(torch.stack([cc_output, mlo_output]), 0).values

        outputs = self.classifier(viewpool)

        return nn.Sigmoid()(outputs)
```

## Train the model
Huggingface also provide a [Trainer function](https://huggingface.co/docs/transformers/main_classes/trainer) that you can borrow and use. This goes hand in hand with the TrainingArguments class which you will have to define to customize your training. Here is a quick example on how to use them:
```
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    f"bert-finetuned-sem_eval-english",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
```
Alternatively, you can also create your own custom training function if you want more flexibility. Normally, the huggingface Trainer function is already very good for common use.

```
def train(model, train_dataloader, val_dataloader, Y_val, path):
    
    best_f1 = 0
    best_loss = 1e5
    no_improvement = 0
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)))
    
    total_loss_train = []
    total_loss_val = []
    f_scores = []
    for epoch in range(num_epochs):
        model.train()
        loss_epoch_train = 0
        for batch in train_dataloader:
            x = batch[0].to(device)
            mask = batch[1].to(device)
            y = torch.reshape(batch[2],  [len(batch[2]),1]).float().to(device)
            outputs = model(x, mask)

            loss = model.loss_fn(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            model.optimizer.step()
            #model.lr_scheduler.step()
            model.optimizer.zero_grad()
            progress_bar.update(1)

            loss_epoch_train += loss.item()

        total_loss_train.append(loss_epoch_train/len(train_dataloader))

        l, f, y_preds = evaluate(model, val_dataloader, Y_val)
        total_loss_val.append(l)
        f_scores.append(f)

        if l < best_loss:
            best_loss = l
            no_improvement = 0
        else:
            no_improvement += 1
            
        if f > best_f1:
            best_f1 = f
            torch.save(model.state_dict(), path)

        print('Train loss:', total_loss_train[-1])
        print('Val loss:', total_loss_val[-1])
      
        # Early stop conditions
        if no_improvement == tolerance:
            print('EARLY STOP! Epoch =', epoch)
            break
    
    model.load_state_dict(torch.load(path))
```
## Putting everything together
Huggingface also provides a lot of good example script for common use case such as [text classification](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb), [question answering](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb) or [multiple choice](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb). They also include a lot of good exaplanation in their articles so don't forget to check the [Resources section](https://huggingface.co/docs/transformers/model_doc/bert#resources)

# Finetuning BERT

# Beyond BERT

Other resources:
1. [BERT 101 by huggingface](https://huggingface.co/blog/bert-101)
