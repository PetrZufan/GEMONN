import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AutoEncoder, self).__init__()
        self.encode_func = nn.Linear(input_size, hidden_size)
        self.decode_func = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x):
        encoded = torch.sigmoid(self.encode_func(x))
        decoded = torch.sigmoid(self.decode_func(encoded))
        return decoded, encoded

    def set_weights(self, encode_weights, encode_biases):
        state_dict = self.state_dict()
        # state_dict['encode_func.weight'] = torch.ones([784, 10], dtype=torch.float32, device="cuda")
        state_dict['encode_func.weight'] = torch.from_numpy(encode_weights)
        state_dict['encode_func.bias'] = torch.from_numpy(encode_biases)
        state_dict['decode_func.weight'] = torch.from_numpy(encode_weights.t())


def train(model, data_loader, data_size):
    with torch.no_grad():
        loss(model, data_loader, data_size)


def model_output(model, data_loader, data_size):
    # data_iter = iter(data_loader)
    # images, labels = data_iter.next()
    for _, (images, labels) in enumerate(data_loader):
        images = images.view(-1, data_size).cuda()
        # labels = labels.cuda()
        return images, model(images)


def loss(model, data_loader, data_size):
    ins, outs = model_output(model, data_loader, data_size)
    loss_fn = nn.MSELoss()
    return loss_fn(ins, outs[0]).item()
