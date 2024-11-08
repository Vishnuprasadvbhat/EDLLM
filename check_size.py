import os

def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

model_path = "C:\\Users\\vishn\\fed_up\\Quantized"  # Your model directory
size_in_bytes = get_directory_size(model_path)
size_in_megabytes = size_in_bytes / (1024 * 1024)

print(f"Model size: {size_in_megabytes:.2f} MB")

# Model size: 1.37 MB _ Earlier model 

# Model size: 55.65 MB Newly sent  lora_model


# Model size: 42.39 MB Quantized 


