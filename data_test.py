import datasets
import pandas as pd

# Load the IMDb dataset
imdb = datasets.load_dataset('imdb')

# Get the training data
train_data = imdb['train']

# Convert the dataset to a pandas DataFrame
df = train_data.to_pandas()

# Display the first 10 rows
print(df.head(10))
