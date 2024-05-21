import torch

class TimeAxis:
    def __init__(self, device):
        self.device = device
        self.time_axis = {}

    def get_current_time(self, key, length, features_num):
        if key not in self.time_axis:
            h_0 = torch.zeros((12, length, features_num), device=self.device)
            self.time_axis[key] = h_0
        return self.time_axis[key]

    def update_time(self, key, h_t):
        self.time_axis[key] = h_t.detach()