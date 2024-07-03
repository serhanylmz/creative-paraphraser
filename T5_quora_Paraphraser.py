#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from datasets import load_dataset
from transformers import get_scheduler
from tqdm.auto import tqdm


# In[2]:


get_ipython().system("tmux display-message -p '#S'")


# ## Load and Process the Dataset

# In[ ]:


dataset = load_dataset("quora")

dataset


# In[ ]:


dataset = dataset.filter(lambda x: x['is_duplicate'] == 1)


# In[ ]:


# Define a function to flatten and prepare the data
def prepare_data(examples):
    # Create lists to store processed examples
    input_texts = []
    target_texts = []
    
    # Process each entry
    for question_pair in examples['questions']:
        # Assuming each entry in 'questions' has two questions
        if len(question_pair['text']) == 2:
            input_texts.append("paraphrase: " + question_pair['text'][0])
            target_texts.append(question_pair['text'][1])
    
    # Return a dictionary of processed examples
    return {'input_text': input_texts, 'target_text': target_texts}

# Apply the function to each entry in the dataset
processed_datasets = dataset.map(prepare_data, batched=True, remove_columns=['questions', 'is_duplicate'])


# In[ ]:


processed_datasets


# ## Tokenize the Data

# In[ ]:


from transformers import T5Tokenizer

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-3b')

# Define the function to tokenize the data
def tokenize_function(examples):
    model_inputs = tokenizer(examples['input_text'], max_length=128, truncation=True, padding="max_length")
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target_text'], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# In[ ]:


# Apply tokenization to all sets in the dataset
tokenized_datasets = processed_datasets.map(tokenize_function, batched=True)


# ## Prepare the Dataloaders

# In[ ]:


from torch.utils.data import DataLoader

# Define a helper function to create the DataLoader
def create_dataloader(tokenized_data, batch_size=8):
    # Convert list of dictionaries into a format DataLoader can handle
    dataset = tokenized_data.remove_columns(['input_text', 'target_text'])  # Remove text columns not needed for training
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Create the DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create DataLoaders for training (and optionally validation)
train_dataloader = create_dataloader(tokenized_datasets['train'])


# ## Load Model / Set Up Training

# In[11]:


model = T5ForConditionalGeneration.from_pretrained('t5-3b').cuda()

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Number of training epochs
num_epochs = 3

# Set up the learning rate scheduler
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)


# ## Train the Model

# In[ ]:


progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


# ## Save the Model

# In[15]:


model.save_pretrained("./t5_paraphrase_model")
tokenizer.save_pretrained("./t5_paraphrase_model")


# # Inference

# ## Load model and tokenizer

# In[3]:


from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the model and tokenizer
model_path = "./t5_paraphrase_model"
model = T5ForConditionalGeneration.from_pretrained(model_path).cuda()
tokenizer = T5Tokenizer.from_pretrained(model_path)


# ## Function to Generate Paraphrases

# In[9]:


def generate_paraphrases(input_text, num_returns=3):
    # Encode the input text
    input_ids = tokenizer.encode("paraphrase: " + input_text, return_tensors="pt").to(model.device)
    
    # Generate paraphrases
    paraphrases = model.generate(
        input_ids,
        max_length=50,
        num_beams=num_returns,
        num_return_sequences=num_returns,
        no_repeat_ngram_size=1,
        early_stopping=True
    )
    
    # Decode and print each paraphrase
    return [tokenizer.decode(paraphrase, skip_special_tokens=True) for paraphrase in paraphrases]


# ## Generate

# In[19]:


# Example usage
input_sentence = "What is the best way to learn artificial intelligence?"
paraphrase_outputs = generate_paraphrases(input_sentence, num_returns=5)
for i, paraphrase in enumerate(paraphrase_outputs, 1):
    print(f"Paraphrase {i}: {paraphrase}")


# In[20]:


# Example usage
input_sentence = "What occupation did Albert Einstein have?"
paraphrase_outputs = generate_paraphrases(input_sentence, num_returns=5)
for i, paraphrase in enumerate(paraphrase_outputs, 1):
    print(f"Paraphrase {i}: {paraphrase}")


# In[21]:


# Example usage
input_sentence = "What nationality did the physicist Albert Einstein have?"
paraphrase_outputs = generate_paraphrases(input_sentence, num_returns=5)
for i, paraphrase in enumerate(paraphrase_outputs, 1):
    print(f"Paraphrase {i}: {paraphrase}")


# In[10]:


# Example usage
input_sentence = "The restaurant is a carved-off space up a couple of stairs to one side, dominated by faux bare-brick columns, faux-wood floors and an air of foetid despondency"
paraphrase_outputs = generate_paraphrases(input_sentence, num_returns=5)
for i, paraphrase in enumerate(paraphrase_outputs, 1):
    print(f"Paraphrase {i}: {paraphrase}")

