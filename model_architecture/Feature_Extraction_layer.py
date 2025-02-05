
# class FeatureExtractionLayer(nn.Module):
#     def __init__(self, input_channels, intermediate_channels, final_channels):
#         super(FeatureExtractionLayer, self).__init__()

#         # Header processing: 1×1 convolutions for column context
#         self.header_conv1 = nn.Conv1d(input_channels, intermediate_channels, kernel_size=1)
#         self.header_activation1 = nn.ReLU()
#         self.header_conv2 = nn.Conv1d(intermediate_channels, final_channels, kernel_size=1)  # Reduce to final channels
#         self.header_activation2 = nn.ReLU()

#         # Data processing: 1×2 and 1×1 convolutions for rows/columns
#         self.data_conv1 = nn.Conv1d(input_channels, intermediate_channels, kernel_size=2, padding=1)  # Row-wise filters
#         self.data_activation1 = nn.ReLU()
#         self.data_conv2 = nn.Conv1d(intermediate_channels, final_channels, kernel_size=1)  # Column-wise filters
#         self.data_activation2 = nn.ReLU()

#         # Average Pooling for down-sampling
#         self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

#         # Final Flattening
#         self.flatten = nn.Flatten()

#     def forward(self, x):
#         if x.ndim != 4:
#             raise ValueError(f"Expected 4D input tensor (batch_size, rows, cols, channels), got {x.ndim}D tensor.")

#         # Separate header and data
#         header = x[:, :1, :, :]  # First row as header, shape: (batch_size, 1, cols, channels)
#         print(f"Header shape: {header.shape}")
#         data = x[:, 1:, :, :]    # Remaining rows as data, shape: (batch_size, 50, cols, channels)
#         print(f"Data shape: {data.shape}")

#         # Process header (no pooling)
#         header = header.squeeze(1).permute(0, 2, 1)  # Shape: (batch_size, channels, cols)
#         header = self.header_conv1(header)
#         print(f"Shape after header conv1: {header.shape}")
#         header = self.header_activation1(header)
#         header = self.header_conv2(header)
#         print(f"Shape after header conv2: {header.shape}")
#         header = self.header_activation2(header)  # Shape: (batch_size, final_channels, cols)

#         # Process data
#         data = data.permute(0, 3, 2, 1).reshape(data.size(0), data.size(3), -1)  # Shape: (batch_size, channels, seq_len)
#         data = self.data_conv1(data)
#         data = self.data_activation1(data)
#         data = self.data_conv2(data)
#         data = self.data_activation2(data)

#         data = self.pool(data)  # Shape: (batch_size, final_channels, reduced_seq_len)

#         # Combine header and data features
#         combined = torch.cat([header, data], dim=2)  # Concatenate along sequence dimension
#         combined = self.flatten(combined)  # Shape: (batch_size, flattened_features)

#         return combined

# class OutputLayer(nn.Module):
#     def __init__(self, input_dim, output_dim=270):
#         super(OutputLayer, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 512)
#         self.fc2 = nn.Linear(512, output_dim)
#         self.activation = nn.ReLU()

#     def forward(self, x):
#         x = self.activation(self.fc1(x))
#         return self.fc2(x)




import torch
import torch.nn as nn


#---------- Normal Avg Pooling ----------------------------------



# class FeatureExtractionLayer(nn.Module):
#     def __init__(self, expected_output_size=1600):
#         super(FeatureExtractionLayer, self).__init__()

#         # *Header Processing Path (32 → 8)*
#         self.header_conv = nn.Conv1d(32, 8, kernel_size=1)  # 32 → 8
#         self.header_activation = nn.ReLU()

#         # *Column Processing Path (32 → 8)*
#         self.column_conv1 = nn.Conv1d(32, 8, kernel_size=1)  # 32 → 8
#         self.column_activation1 = nn.ReLU()
#         self.column_pool = nn.AvgPool1d(kernel_size=2, stride=2)  # Pooling

#         # *Row Processing Path (32 → 8)*
#         self.row_conv1 = nn.Conv1d(32, 8, kernel_size=1)  # 32 → 8
#         self.row_activation1 = nn.ReLU()
#         self.row_pool = nn.AvgPool1d(kernel_size=2, stride=2)  # Pooling

#         # *Flattening*
#         self.expected_output_size = expected_output_size
#         self.flatten = nn.Flatten()

#     def forward(self, x):
#         print(f"Before Processing in FeatureExtractionLayer: {x.shape}") 

#         batch_size, channels, height, width = x.shape 

#         header = x[:, :, :, 0]  
#         print(f"Header Shape Before Conv1D: {header.shape}")  

#         header = header.permute(0, 1, 2)  
#         header = self.header_activation(self.header_conv(header))  
#         print(f"Header Shape After Conv1D: {header.shape}") 

        
        
#         # Compute width dynamically
#         batch_size, channels, height, width = x.shape  

#         # Permute the tensor to rearrange dimensions
#         columns = x.permute(0, 2, 3, 1) 
#         print(f"Column Shape Before flattening: {columns.shape}")
        
#         columns = columns.reshape(batch_size, 32, -1)
#         print(f"Fixed Column Shape Before Conv1D: {columns.shape}")  
        
#         new_width = columns.shape[-1]
#         print(f"New width after flattening: {new_width}")
#         columns = columns.reshape(batch_size, 32, new_width)
#         print(f"Column Shape Before flattening: {columns.shape}")
      
#         columns = self.column_activation1(self.column_conv1(columns))
#         columns = nn.AvgPool1d(kernel_size=101, stride=101, count_include_pad=False)(columns) 
#         print(f"Column Shape After Conv1D: {columns.shape}") 


#         rows = x.permute(0, 1, 3, 2)  
#         print(f"Row Shape Before Flattening: {rows.shape}") 
#         rows = rows.reshape(batch_size, 32, -1)
#         print(f"Fixed Row Shape Before Conv1D: {rows.shape}")  
#         new_height= rows.shape[-1]
#         print(f"New height after flattening: {new_height}")
#         rows = rows.reshape(batch_size, 32, new_height)
#         print(f"Fixed Row Shape Before Conv1D: {rows.shape}")  

#         rows = self.row_activation1(self.row_conv1(rows))
#         rows = nn.AvgPool1d(kernel_size=50, stride=50, count_include_pad=False)(rows)  
#         print(f"Row Shape After Conv1D: {rows.shape}")

#         # *Concatenate Header with Columns & Rows*
#         combined_features = torch.cat([header, columns, rows], dim=2)  # Header + Columns + Rows
#         print(f"Combined Features Shape Before Flattening: {combined_features.shape}") 
#         # *Flatten Before Output*
#         output_features = self.flatten(combined_features)
#         print(f"Extracted Features Shape: {output_features.shape}")  

#         return output_features




#-------------- Adaptive Max Pooling to handle target size -------------------------------

class FeatureExtractionLayer(nn.Module):
    def __init__(self):
        super(FeatureExtractionLayer, self).__init__()

        # Header Processing Path (32 → 8)
        self.header_conv = nn.Conv1d(32, 8, kernel_size=1)  # 32 → 8
        self.header_activation = nn.ReLU()

        # Column Processing Path (32 → 64 → 8)
        self.column_conv1 = nn.Conv1d(32, 64, kernel_size=1)  # 32 → 64
        self.column_activation1 = nn.ReLU()
        self.column_conv2 = nn.Conv1d(64, 8, kernel_size=1)   # 64 → 8
        self.column_activation2 = nn.ReLU()
        self.column_pool = nn.AdaptiveMaxPool1d(50)  # Dynamic Pooling

        # Row Processing Path (32 → 64 → 8)
        self.row_conv1 = nn.Conv1d(32, 64, kernel_size=1)  # 32 → 64
        self.row_activation1 = nn.ReLU()
        self.row_conv2 = nn.Conv1d(64, 8, kernel_size=1)   # 64 → 8
        self.row_activation2 = nn.ReLU()
        self.row_pool = nn.AdaptiveMaxPool1d(100)  # Dynamic Pooling

        # Flattening
        self.flatten = nn.Flatten()

    def forward(self, x):
        print(f"Before Processing in FeatureExtractionLayer: {x.shape}")

        batch_size, channels, height, width = x.shape

        # Header Processing
        header = x[:, :, :, 0]
        print(f"Header Shape Before Conv1D: {header.shape}")
        header = header.permute(0, 1, 2)
        print(f"Header Shape after permunation: {header.shape}")  
        header = self.header_activation(self.header_conv(header))  
        print(f"Header Shape After Conv1D: {header.shape}")  

        # Column Processing
        columns = x.permute(0, 2, 3, 1) 
        print(f"Column Shape Before Flattening: {columns.shape}") 
        columns = columns.reshape(batch_size, 32, height*width)
        print(f"Column Shape Before Conv1D: {columns.shape}")

        columns = self.column_activation1(self.column_conv1(columns))
        columns = self.column_activation2(self.column_conv2(columns))
        columns = self.column_pool(columns)
        print(f"Column Shape After Conv1D: {columns.shape}")

        # Row Processing
        rows = x.permute(0, 1, 3, 2)
        print(f"Row Shape Before Flattening: {rows.shape}")  
        rows = rows.reshape(batch_size, 32, height*width)
        print(f"Row Shape Before Conv1D: {rows.shape}")

        rows = self.row_activation1(self.row_conv1(rows))
        rows = self.row_activation2(self.row_conv2(rows))
        rows = self.row_pool(rows)
        print(f"Row Shape After Conv1D: {rows.shape}")

        # Feature Combination
        combined_features = torch.cat([header, columns, rows], dim=2)
        print(f"Combined Features Shape Before Flattening: {combined_features.shape}")

        # Flatten
        output_features = self.flatten(combined_features)
        print(f"Extracted Features Shape: {output_features.shape}")

        return output_features


if __name__ == "__main__":
    
    # input output shape from the Dimension reduction layer 
    input_tensor = torch.rand(1, 32, 101, 50)  # Shape: (batch_size, rows, cols, channels)

    input_reduced_tensor = input_tensor.permute(0, 1, 3, 2)
    print(f"Input Reduced Tensor Shape: {input_reduced_tensor.shape}")
    feature_extraction = FeatureExtractionLayer()
    features = feature_extraction(input_reduced_tensor)
    print("Extracted Features Shape:", features.shape)
