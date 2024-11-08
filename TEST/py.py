import datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Define a function to tokenize the dataset
def preprocess_function(examples, tokenizer, max_length=128):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)

# Tokenize and create PyTorch DataLoader
def create_dataloader(dataset, tokenizer, batch_size=16, shuffle=True, max_length=128):
    # Check the type of the dataset to ensure it's a Hugging Face Dataset object
    if not isinstance(dataset, datasets.arrow_dataset.Dataset):
        raise TypeError(f"Expected a Hugging Face Dataset object, but got {type(dataset)}")

    # Apply tokenization to the dataset
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer, max_length), batched=True)

    # Convert tokenized dataset to PyTorch tensors
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Create a DataLoader
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# Load the IMDb dataset using the datasets module
imdb_dataset = datasets.load_dataset("imdb")

# Separate the dataset into train and test sets
train_data = imdb_dataset['train']
test_data = imdb_dataset['test']

# Initialize the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

# # Create DataLoaders
# train_loader = create_dataloader(train_data, tokenizer=tokenizer)
# test_loader = create_dataloader(test_data, tokenizer=tokenizer)

# # Check types of train_loader and test_loader
# print(type(train_loader))
# print(type(test_loader))

print(type(train_data))