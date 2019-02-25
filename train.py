from argparse import ArgumentParser
import torch as th
import torch.optim as optim
from gan import Discriminator, Generator
import data


parser = ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--data-args', nargs='+', type=int)
parser.add_argument('--dbs', type=int, help='Discriminator Batch Size')
parser.add_argument('--dlr', type=float, help='Discriminator Learning Rate')
parser.add_argument('--glr', type=float, help='Generator Learning Rate')
parser.add_argument('--gbs', type=int, help='Generator Batch Size')
parser.add_argument('--gpu', type=int, help='GPU')
parser.add_argument('--ni', type=int, help='Number of Iterations')
args = parser.parse_args()

device = th.device('cpu') if args.gpu < 0 else None  # TODO
data_loader = getattr(data, args.data + '_loader')(*args.data_args, device=device)
discriminator = Discriminator().to(device)
g_configs = [{'in_feats' : args.n_nidden,
              'out_feats' : args.n_hidden,
              'out_nodes' :
              'aggregator' : 'mean'} for i in range(n_layers)]
generator = Generator().to(device)

d_optim = optim.Adam(discirminator.parameters(), args.dlr)
g_optim = optim.Adam(generator.parameters(), args.glr)

for i in range(args.ni):
    for j in range(args.gbs):
        generator()
    x, adj = next(data_loader)
    p, cost = discriminator(x, adj)
    authentic =
    synthetic = 
