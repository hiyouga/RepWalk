import os
import math
import torch
import random
import argparse
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
from model import RepWalk
from loss_func import CrossEntropy
from data_utils import MyDataset, build_tokenizer, build_embedding_matrix

class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt): # prepare for training the model
        self.opt = opt # hyperparameters and options
        opt.tokenizer = build_tokenizer(fnames=opt.dataset_file.values(), dataset=opt.dataset) # transfrom tokens to indices
        embedding_matrix = build_embedding_matrix(vocab=opt.tokenizer.vocab['word'], dataset=opt.dataset) # pre-trained glove embeddings
        self.trainset = MyDataset(fname=opt.dataset_file['train'], tokenizer=opt.tokenizer) # training set
        self.testset = MyDataset(fname=opt.dataset_file['test'], tokenizer=opt.tokenizer) # testing set
        self.model = RepWalk(embedding_matrix, opt).to(opt.device) # neural network model
        self._print_args() # print arguments
    
    def _print_args(self): # pring arguments
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        if self.opt.device.type == 'cuda':
            print(f"cuda memory allocated: {torch.cuda.memory_allocated(self.opt.device.index)}")
        print(f"n_trainable_params: {int(n_trainable_params)}, n_nontrainable_params: {int(n_nontrainable_params)}")
        print('training arguments:')
        for arg in vars(self.opt):
            print(f">>> {arg}: {getattr(self.opt, arg)}")
    
    def _reset_params(self): # reset model parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'embedding' in name: # treat embedding matrices as special cases
                    weight = torch.nn.init.xavier_uniform_(torch.zeros_like(param)) # use xavier_uniform to initialize embedding matrices
                    weight[0] = torch.tensor(0, dtype=param.dtype, device=param.device) # the vector corresponding to padding index shuold be zero
                    setattr(param, 'data', weight) # update embedding matrix
                else:
                    if len(param.shape) > 1:
                        torch.nn.init.xavier_uniform_(param) # use xavier_uniform to initialize weight matrices
                    else:
                        stdv = 1. / math.sqrt(param.size(0))
                        torch.nn.init.uniform_(param, a=-stdv, b=stdv) # use uniform to initialize bias vectors
    
    def _train(self, dataloader, criterion, optimizer): # train the model
        train_loss, n_correct, n_train = 0, 0, 0 # reset counters
        self.model.train() # switch model to training mode
        for sample_batched in dataloader: # mini-batch optimization
            inputs = list(map(lambda x: x.to(self.opt.device), sample_batched[0])) # move tensors to target device (e.g. cuda)
            labels = sample_batched[1].to(self.opt.device) # move labels to target device
            outputs = self.model(inputs) # compute outputs
            
            optimizer.zero_grad() # clear gradient accumulators
            loss = criterion(outputs, labels) # compute batch loss
            loss.backward() # compute gradients through back-propagation
            optimizer.step() # update model parameters
            
            train_loss += loss.item() * len(labels) # update train loss
            n_correct += (torch.argmax(outputs[0], -1) == labels).sum().item() # update correct sample number
            n_train += len(labels) # update train sample number
        return train_loss / n_train, n_correct / n_train
    
    def _evaluate(self, dataloader, criterion): # evaluate the model
        test_loss, n_correct, n_test = 0, 0, 0 # reset counters
        labels_all, predicts_all = None, None # initialize variables
        self.model.eval() # switch model to evaluation mode
        with torch.no_grad(): # turn off gradients
            for sample_batched in dataloader:
                inputs = list(map(lambda x: x.to(self.opt.device), sample_batched[0]))
                labels = sample_batched[1].to(self.opt.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * len(outputs)
                n_correct += (torch.argmax(outputs[0], -1) == labels).sum().item()
                n_test += len(labels)
                labels_all = torch.cat((labels_all, labels), dim=0) if labels_all is not None else labels
                predicts_all = torch.cat((predicts_all, outputs[0]), dim=0) if predicts_all is not None else outputs[0]
        f1 = metrics.f1_score(labels_all.cpu(), torch.argmax(predicts_all, -1).cpu(), labels=[0, 1, 2], average='macro') # compute f1 score
        return test_loss / n_test, n_correct / n_test, f1
    
    def run(self):
        _params = filter(lambda p: p.requires_grad, self.model.parameters()) # trainable parameters
        optimizer = torch.optim.Adam(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg) # use the Adam optimizer
        criterion = CrossEntropy(beta=self.opt.beta, eps=self.opt.eps) # loss function implemented as described in paper
        
        train_dataloader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True) # training dataloader
        test_dataloader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False) # testing dataloader
        
        self._reset_params() # reset model parameters
        best_test_acc, best_test_f1 = 0, 0 # record the best acc and f1 score on testing set
        for epoch in range(self.opt.num_epoch):
            train_loss, train_acc = self._train(train_dataloader, criterion, optimizer) # train the model
            test_loss, test_acc, test_f1 = self._evaluate(test_dataloader, criterion) # evaluate the model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_f1 = test_f1
            print(f"{100*(epoch+1)/self.opt.num_epoch:6.2f}% > loss: {train_loss:.4f}, acc: {train_acc:.4f}, test acc: {test_acc:.4f}, test f1: {test_f1:.4f}")
        print('#' * 50)
        print(f"best test acc: {best_test_acc:.4f}, best test f1: {best_test_f1:.4f}")
    

def main():
    ''' dataset files'''
    dataset_files = {
        'restaurant': {
            'train': os.path.join('datasets', 'Restaurants_Train.json'),
            'test': os.path.join('datasets', 'Restaurants_Test.json')
        },
        'laptop': {
            'train': os.path.join('datasets', 'Laptops_Train.json'),
            'test': os.path.join('datasets', 'Laptops_Test.json')
        },
        'twitter': {
            'train': os.path.join('datasets', 'Tweets_Train.json'),
            'test': os.path.join('datasets', 'Tweets_Test.json')
        },
        'restaurant16': {
            'train': os.path.join('datasets', 'Restaurants16_Train.json'),
            'test': os.path.join('datasets', 'Restaurants16_Test.json')
        }
    }
    ''' hyperparameters '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='restaurant', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--word_dim', default=300, type=int)
    parser.add_argument('--pos_dim', default=30, type=int)
    parser.add_argument('--dep_dim', default=50, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--embed_dropout', default=0.5, type=float)
    parser.add_argument('--bilinear_dropout', default=0, type=float)
    parser.add_argument('--fc_dropout', default=0, type=float)
    parser.add_argument('--beta', default=0.01, type=float)
    parser.add_argument('--eps', default=0.01, type=float)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    ''' parse arguments '''
    opt = parser.parse_args()
    opt.dataset_file = dataset_files[opt.dataset]
    opt.seed = opt.seed if opt.seed else random.randint(0, 4294967295)
    opt.device = torch.device(opt.device) if opt.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ''' set random seed '''
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    ''' if you are using cudnn '''
    torch.backends.cudnn.deterministic = True # Deterministic mode can have a performance impact
    torch.backends.cudnn.benchmark = False
    ''' run the model '''
    ins = Instructor(opt)
    ins.run()

if __name__ == '__main__':
    main()
