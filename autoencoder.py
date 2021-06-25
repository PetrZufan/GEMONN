import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AutoEncoder, self).__init__()
        self.encode_func = nn.Linear(input_size, hidden_size)
        self.decode_func = nn.Linear(hidden_size, output_size, bias=False)

    def set_data(self, data_loader, data_size):
        self.data_loader = data_loader
        self.data_size = data_size

    def forward(self, x):
        encoded = torch.sigmoid(self.encode_func(x))
        decoded = torch.sigmoid(self.decode_func(encoded))
        return decoded, encoded

    def set_weights(self, encode_weights, encode_biases):
        state_dict = self.state_dict()
        # state_dict['encode_func.weight'] = torch.ones([784, 10], dtype=torch.float32, device=device)
        state_dict['encode_func.weight'] = torch.from_numpy(encode_weights)
        state_dict['encode_func.bias'] = torch.from_numpy(encode_biases)
        state_dict['decode_func.weight'] = torch.from_numpy(encode_weights).t()

    def get_weights(self):
        return self.encode_func.weight, self.encode_func.bias

    def evaluate(self):
        self.eval()
        with torch.no_grad():
            return self.loss(self.data_loader, self.data_size)

    def loss(self, data_loader, data_size):
        ins, outs = self.output(data_loader, data_size)
        loss_fn = nn.MSELoss()
        return loss_fn(ins, outs[0]).item()

    def output(self, data_loader, data_size):
        # data_iter = iter(data_loader)
        # images, labels = data_iter.next()
        for _, (images, labels) in enumerate(data_loader):
            images = images.view(-1, data_size).to(device)
            # labels = labels.to(device)
            return images, self(images)
