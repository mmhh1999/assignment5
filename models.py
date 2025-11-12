import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        # A small PointNet-like classifier
        # Per-point MLP (implemented as 1D convs) to extract point features
        self.feat = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        # MLP for classification from global feature
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # points: (B, N, 3) -> (B, 3, N)
        x = points.permute(0, 2, 1).contiguous()
        x = self.feat(x)            # (B, 1024, N)
        x = torch.max(x, 2)[0]     # global max-pool -> (B, 1024)
        x = self.classifier(x)     # (B, num_classes)
        return x



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        # Per-point feature extractor
        self.local_feat = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Deeper per-point -> global
        self.global_feat = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
        )

        # Per-point classifier that uses local + global features
        # input channels: 128 (local) + 1024 (global expanded)
        self.seg_head = nn.Sequential(
            nn.Conv1d(128 + 1024, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, num_seg_classes, 1),
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        # points: (B, N, 3) -> (B, 3, N)
        x = points.permute(0, 2, 1).contiguous()

        local = self.local_feat(x)      # (B, 128, N)
        g = self.global_feat(local)     # (B, 1024, N)
        gpool = torch.max(g, 2, keepdim=True)[0]  # (B, 1024, 1)
        g_expanded = gpool.repeat(1, 1, local.size(2))  # (B, 1024, N)

        concat = torch.cat([local, g_expanded], dim=1)  # (B, 128+1024, N)
        seg_logits = self.seg_head(concat)  # (B, num_seg_classes, N)

        # return (B, N, num_seg_classes)
        seg_logits = seg_logits.permute(0, 2, 1).contiguous()
        return seg_logits



