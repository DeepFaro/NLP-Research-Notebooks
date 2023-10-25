"""
install required packages on terminal

!pip install transformers
!pip install sentencepiece
"""

import torch
import pandas as pd
from torch.utils.data import Dataset, random_split
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, T5Tokenizer

# Dataset class
class BeautyDataset(Dataset):
    def __init__(self, txt_list, label_list, tokenizer, max_length):
        # define variables    
        self.input_ids = []
        self.attn_masks = []
        #self.labels = []
        
        # iterate through the dataset
        # truncate long content > 256
        for txt, label in zip(txt_list, label_list):
            txt= txt.replace("\n", "")
            txt_ = (txt[:256]) if len(txt) > 256 else txt

            # prepare the text
            prep_txt = f'<s>Content: {txt_}[SEP]Title: {label}</s>'

            # tokenize
            encodings_dict = tokenizer(prep_txt, truncation=True,
                                       max_length=max_length, padding="max_length")
            # append to list
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            #self.labels.append(label)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


def read_content_title():
    import glob

    path = r'all' # folder
    all_files = glob.glob(path + "/*.csv")

    list_ = []
    for filename in all_files:
        try:
            #print(filename)
            #skipe 1st row
            df_t = pd.read_csv(filename, index_col=None, header=None, skiprows=1, encoding='utf-8')
            list_.append(df_t)
        except:
            print(f"reading error in{filename}")

    frame = pd.concat(list_, axis=0, ignore_index=True)
    print(f"length of data frame: {len(frame)}")
    return frame

# Data load function
def load_beauty_dataset(tokenizer):
    # load dataset and sample.
    df = read_content_title()
    df = df[[0, 1]]
    df.columns = ['content', 'title']
    #df = df.sample(20000, random_state=1)
    
    max_length = max([len(tokenizer.encode(description)) for description in df['content']])
    print("Max length: {}".format(max_length))

    dataset = BeautyDataset(df['content'].tolist(), df['title'].tolist(), tokenizer, max_length=512)
    dataset.__getitem__(5)

    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    print(len(dataset))

    # return
    return train_dataset, val_dataset


#load model
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium", bos_token='<s>', eos_token='</s>', pad_token='<pad>')
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium").cuda()

model.resize_token_embeddings(len(tokenizer))

print("Loading dataset...")
train_dataset, val_dataset = load_beauty_dataset(tokenizer)

print("Start training...")
training_args = TrainingArguments(output_dir=r'result-content-title', num_train_epochs=5, 
                                logging_steps=5000, load_best_model_at_end=True,
                                save_strategy='steps',
                                evaluation_strategy="steps",
                                save_steps=10000,
                                per_device_train_batch_size=12, per_device_eval_batch_size=12,
                                learning_rate=0.01,
                                warmup_steps=2000, weight_decay=0.0001, logging_dir='logs')


trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
          eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                'attention_mask': torch.stack([f[1] for f in data]),
                                                                'labels': torch.stack([f[0] for f in data])})
trainer.train() 

output_dir = 'result-content-title/final'
# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)