import torch

# Initial tensor with shape (batch, sequence_length, hidden_size)
batch = 32
sequence_length = 10
hidden_size = 768
tensor = torch.randn(batch, sequence_length, hidden_size)

# Reshape to (batch*sequence_length, hidden_size)
reshaped_tensor = tensor.reshape(batch*sequence_length, hidden_size)

# Calculate the original sequence_length for conversion back
total_elements = reshaped_tensor.numel()
calculated_sequence_length = total_elements // (batch * hidden_size)

# Convert back to the original shape
original_shape_tensor = reshaped_tensor.view(batch, calculated_sequence_length, hidden_size)

print("Original shape:", tensor.shape)
print("Reshaped shape:", reshaped_tensor.shape)
print("Converted back shape:", original_shape_tensor.shape)