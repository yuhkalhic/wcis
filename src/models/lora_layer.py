import torch

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, time_axis, layer_key, device):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.W_a = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.W_b = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.W_c = torch.nn.Parameter(torch.zeros(in_dim, out_dim))
        self.W_d = torch.nn.Parameter(torch.zeros(in_dim, out_dim))
        self.alpha = alpha
        self.time_axis = time_axis
        self.layer_key = layer_key
        self.device = device
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        h_t = self.time_axis.get_current_time(self.layer_key, x.size()[1], x.size()[2])
        # print("h_t size:", h_t.size())
        # print("x size:", x.size())
        # print("W_c size:", self.W_c.size())
        # print("W_d size:", self.W_d.size())

        if h_t.size()[0] != x.size()[0]:
            if h_t.size()[0] < x.size()[0]:
                diff = x.size()[0] - h_t.size()[0]
                half_padding = torch.full((diff, h_t.size()[1], h_t.size()[2]), 0.5, device=self.device)
                h_t = torch.cat([h_t, half_padding], dim=0)
            else:
                indices = torch.randperm(h_t.size()[0])[:x.size()[0]]
                h_t = h_t[indices]

        h_t1 = (h_t @ self.W_c + x @ self.W_d) + h_t
        min_val = h_t1.min(dim=2, keepdim=True)[0]
        max_val = h_t1.max(dim=2, keepdim=True)[0]
        h_t1_normed = (h_t1 - min_val) / (max_val - min_val + 1e-8)
        self.time_axis.update_time(self.layer_key, h_t1_normed.detach())
        y = self.alpha * (x @ self.W_a @ self.W_b) + torch.nn.functional.leaky_relu(self.dropout(h_t1_normed))
        return y


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha, time_axis, layer_key, device):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha, time_axis, layer_key, device
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)