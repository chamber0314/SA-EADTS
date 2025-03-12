import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
import torch


class encoder(nn.Module):
    def __init__(self, feature_dim,kernel_size, hidden,domainNum, type="ori"):
        super(encoder, self).__init__()
        self.bn = nn.BatchNorm1d(hidden, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        # self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.bottleneck = nn.Linear(feature_dim, hidden)
        self.bottleneck1 = nn.Linear(hidden, 256)
        self.fc = nn.Linear(256, domainNum)
        self.classify = nn.Linear(256, 2)
        self.decoder = nn.Linear(256,feature_dim)
        self.type = type


    def init_cluster(self, loader, num_sample=5000, num_class=0):
        data = loader.dataset.data
        from sklearn.cluster import KMeans
        n,t = data.shape
        ids = torch.randint(low=0, high=n, size=(num_sample,))
        pos = torch.randint(low=0, high=t - self.kernel_size, size=(num_sample,))
        presamples = loader.dataset[ids][0] # shape  (num_sample,t)
        samples = []
        for i in range(num_sample):
            samples.append(presamples[i][pos[i]:pos[i] + self.kernel_size])
        samples = torch.stack(samples)
        kmeans = KMeans(n_clusters=self.dim).fit(samples.numpy())
        cluster_centers = torch.tensor(kmeans.cluster_centers_)
        self.conv.weight = torch.nn.Parameter(cluster_centers.unsqueeze(1))#

    def forward(self, x):
        b,h,w = x.shape
        x = x.contiguous()
        x = x.view(b,h*w)
        x = self.bottleneck(x)
        x = self.relu(x)
        x = self.bottleneck1(x)
        x = self.relu(x)
        x_new = self.decoder(x)
        cls_x = self.fc(x)
        classify = self.classify(x)
        if self.type == "bn":
            x = self.bn(x)

        return x,cls_x,x_new,classify


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(
                nn.Linear(bottleneck_dim, class_num), name="weight")
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x
