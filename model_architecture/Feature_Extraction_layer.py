import torch
import torch.nn as nn

class FeatureExtractionLayer(nn.Module):
    def __init__(self, in_channels=32, mid_channels=64, out_channels=8):
        super(FeatureExtractionLayer, self).__init__()

        # ==== Column Feature Extraction ====
        self.conv1x1_col1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1))
        self.conv1x2_col1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 2), padding=(0, 1))
        self.avgpool_col1 = nn.AvgPool2d(kernel_size=(100, 1), stride=(100, 1))

        self.conv1x1_col2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1))
        self.conv1x2_col2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 2), padding=(0, 1))

        # ==== Row Feature Extraction ====
        self.conv1x1_row1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1))
        self.conv1x2_row1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(2, 1), padding=(1, 0))
        self.avgpool_row1 = nn.AvgPool2d(kernel_size=(1, 50), stride=(1, 50))

        self.conv1x1_row2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(1, 1))
        self.conv1x2_row2 = nn.Conv2d(mid_channels, out_channels, kernel_size=(2, 1), padding=(1, 0))

        # ==== Header Feature Extraction ====
        self.conv1x1_header = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size, _, height, width = x.shape

        # ====== Header Extraction ======
        header = x[:, :, 0:1, :]  # ✅ Extract first row
        x = x[:, :, 1:, :]
        print(f"header shape after permute: {header.shape}")
        header_features = self.activation(self.conv1x1_header(header))  # ✅ Apply `1×1` filter
        print(f"Header Shape: {header_features.shape}")

        # ====== Column Feature Extraction ======
        
        # Apply 1x1 convolution
        global_col1 = self.activation(self.conv1x1_col1(x)) 
        
        # Apply 1x2 convolution 
        local_col1 = self.activation(self.conv1x2_col1(x))   

       
        min_width = min(global_col1.shape[3], local_col1.shape[3])
        
        # crop the min global width to match local width
        global_col1 = global_col1[:, :, :, :min_width]
        
        # crop the min local width to match global width
        local_col1 = local_col1[:, :, :, :min_width]

        # Apply average pooling
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
        # crop the min global height to match local height
        global_row1 = global_row1[:, :, :min_height, :]
        # crop the min local height to match global height
        local_row1 = local_row1[:, :, :min_height, :]

        global_row1_pooled = self.avgpool_row1(global_row1)  
        local_row1_pooled = self.avgpool_row1(local_row1)    

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
        header_features_flat = header_features.view(batch_size, -1) 
        print(f"Header Features Shape : {header_features_flat.shape}")

        combined_features = torch.cat([column_features_flat, row_features_flat, header_features_flat], dim=1)

        return combined_features

if __name__ == "__main__":
    input_tensor = torch.rand(1, 32, 101, 50)
    feature_extractor = FeatureExtractionLayer()
    output = feature_extractor(input_tensor)
    print("✅ Final Output Shape:", output.shape)  
