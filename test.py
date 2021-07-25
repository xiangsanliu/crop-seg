
from model.setr.SETR import SETR_Naive
import torch
from torchsummary import summary




def main():
    net = SETRModel(config)
    print(net)
    t1 = torch.rand(2, 3, 512, 512)
    t1 = net(t1)
    print(t1.shape)

if __name__ == '__main__':
    main()