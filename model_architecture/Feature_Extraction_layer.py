import torch
import torch.nn as nn

class FeatureExtractionLayer(nn.Module):
    def __init__(self, in_channels=32, mid_channels=64, out_channels=8):
        super(FeatureExtractionLayer, self).__init__()

        # ==== Column Feature Extraction ====
        
        # apply the 1x1 and 1x2 convolution to increse the dimension 32 → 64
        self.conv1x1_col1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1))
        self.conv1x2_col1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 2), padding=(0, 1))
        
        # Applied average pooling across all rows for each column
        self.avgpool_col1 = nn.AvgPool2d(kernel_size=(100, 1), stride=(100, 1))

        # apply the 1x1 and 1x2 convolution again to reduce the dimensions 64 → 8
        self.conv1x1_col2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1))
        self.conv1x2_col2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 2), padding=(0, 1))


        # ==== Row Feature Extraction ====
        
        # apply the 1x1 and 1x2 convolution to increase the dimension 32 → 64
        self.conv1x1_row1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1))
        self.conv1x2_row1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(2, 1), padding=(1, 0)) # Convolution applied  2x1 filter, since rows are arranged top-to-bottom, so this filter capture dependencies between adjacent rows rows
        
        # Applied average pooling across all columns for each row
        self.avgpool_row1 = nn.AvgPool2d(kernel_size=(1, 50), stride=(1, 50))

        # apply the 1x1 and 1x2 convolution again to reduce the dimensions 64 → 8
        self.conv1x1_row2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1))
        self.conv1x2_row2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(2, 1), padding=(1, 0)) # Convolution applied  2x1 filter, since rows are arranged top-to-bottom, so this filter capture dependencies between adjacent rows rows

        # ==== Header Feature Extraction ====
        
        # apply the 1x1 and  1x2 convolution to reduce the dimensions 32 → 8
        self.conv1x1_header = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.conv1x2_header = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 2), padding=(0, 1))

        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size, _, height, width = x.shape

        # ====== Header Extraction ======
        header = x[:, :, 0:1, :]  # Extract first row
        x = x[:, :, 1:, :]
        
        header_features_1 = self.activation(self.conv1x1_header(header))
        header_features_2 = self.activation(self.conv1x2_header(header))
        
        min_width_header = min(header_features_1.shape[3], header_features_2.shape[3])
        header_features_1 = header_features_1[:, :, :, :min_width_header]
        header_features_2 = header_features_2[:, :, :, :min_width_header]
        
        header = header_features_1 + header_features_2
        print(f"Header Shape after concatenation: {header.shape}")

        # ====== Column Feature Extraction ======
        global_col1 = self.activation(self.conv1x1_col1(x)) 
        local_col1 = self.activation(self.conv1x2_col1(x))   
        min_width = min(global_col1.shape[3], local_col1.shape[3])
        global_col1 = global_col1[:, :, :, :min_width]
        local_col1 = local_col1[:, :, :, :min_width]

        global_col1_pooled = self.avgpool_col1(global_col1)  
        local_col1_pooled = self.avgpool_col1(local_col1)    

        column_features1 = global_col1_pooled + local_col1_pooled  
        print(f"Column Features Shape after Pooling: {column_features1.shape}")
        global_col2 = self.activation(self.conv1x1_col2(column_features1))  
        local_col2 = self.activation(self.conv1x2_col2(column_features1))   

        min_width2 = min(global_col2.shape[3], local_col2.shape[3])
        global_col2 = global_col2[:, :, :, :min_width2]
        local_col2 = local_col2[:, :, :, :min_width2]

        column_features2 = global_col2 + local_col2 
        print(f"Column Features Shape second betch of filter: {column_features2.shape}")

        # ====== Row Feature Extraction ======
        global_row1 = self.activation(self.conv1x1_row1(x))  
        local_row1 = self.activation(self.conv1x2_row1(x))   

        min_height = min(global_row1.shape[2], local_row1.shape[2])
        global_row1 = global_row1[:, :, :min_height, :]
        print(f"Global Row1 Shape: {global_row1.shape}")
        local_row1 = local_row1[:, :, :min_height, :]
        print(f"Local Row1 Shape: {local_row1.shape}")

        global_row1_pooled = self.avgpool_row1(global_row1)  
        print(f"Global Row1 Pooled Shape: {global_row1_pooled.shape}")
        local_row1_pooled = self.avgpool_row1(local_row1)  
        print(f"Local Row1 Pooled Shape: {local_row1_pooled.shape}")  

        row_features1 = global_row1_pooled + local_row1_pooled  

        global_row2 = self.activation(self.conv1x1_row2(row_features1))  
        local_row2 = self.activation(self.conv1x2_row2(row_features1))   

        min_height2 = min(global_row2.shape[2], local_row2.shape[2])
        global_row2 = global_row2[:, :, :min_height2, :]
        local_row2 = local_row2[:, :, :min_height2, :]

        row_features2 = global_row2 + local_row2  
        print(f"Row Features Shape: {row_features2.shape}")


        # ====== Flatten and Combine Features ======
        column_features_flat = column_features2.view(batch_size, -1)  
        print(f"Column Features Shape: {column_features_flat.shape}")
        row_features_flat = row_features2.view(batch_size, -1)        
        print(f"row_features_flat.shape", row_features_flat.shape)
        header_features_flat = header.view(batch_size, -1) 
        print(f"Header Features Shape : {header_features_flat.shape}")

        combined_features = torch.cat([column_features_flat, row_features_flat, header_features_flat], dim=1)

        return combined_features

if __name__ == "__main__":
    input_tensor = torch.rand(1, 32, 101, 50)
    feature_extractor = FeatureExtractionLayer()
    output = feature_extractor(input_tensor)
    print("Final Output Shape:", output.shape)  
