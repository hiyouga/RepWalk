import torch
import torch.nn as nn
from layer import DynamicLSTM

class RepWalk(nn.Module):
    ''' Neural Network Structure '''
    def __init__(self, embedding_matrix, opt):
        super(RepWalk, self).__init__() # initialize the super class
        ''' common variables '''
        WD = opt.word_dim # dimension of word embeddings
        PN = len(opt.tokenizer.vocab['pos']) # number of pos tags in vocabulary
        PD = opt.pos_dim # dimension of pos tag embeddings
        P_PAD = opt.tokenizer.vocab['pos'].pad_id # padding index of pog tags
        RN = len(opt.tokenizer.vocab['deprel']) # number of dependency relation in vocabulary
        RD = opt.dep_dim # dimension of dependency relation embeddings
        R_PAD = opt.tokenizer.vocab['deprel'].pad_id # padding index of dependency relation
        HD = opt.hidden_dim # dimension of bi-gru hidden state
        ''' embedding layer '''
        self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float)) # pre-trained embedding layer
        self.pos_embedding = nn.Embedding(PN, PD, padding_idx=P_PAD) # pos tag embedding layer
        self.deprel_embedding = nn.Embedding(RN, RD, padding_idx=R_PAD) # dependency relation embedding layer
        ''' other parameter '''
        self.pad_word = nn.Parameter(torch.zeros(HD*2), requires_grad=False) # the padding word for training
        self.pad_edge = nn.Parameter(torch.ones(1), requires_grad=False) # the padding edge for training
        self.ext_rel = nn.Parameter(torch.Tensor(RD), requires_grad=True) # embedding for the edge with 'ext' relation
        ''' main layer '''
        self.rnn = DynamicLSTM(WD+PD, HD, num_layers=1, batch_first=True, bidirectional=True, rnn_type='GRU') # bi-gru layer
        self.bilinear = nn.Bilinear(HD*4, RD, 1) # bilinear layer for score function
        self.fc_out = nn.Linear(HD*2, 3) # fully-connected output layer
        ''' dropout layer '''
        self.embed_dropout = nn.Dropout(opt.embed_dropout) # dropout for embeddings
        self.bilinear_dropout = nn.Dropout(opt.bilinear_dropout) # dropout for bilinear layer
        self.fc_dropout = nn.Dropout(opt.fc_dropout) # dropout for fully-connected layer
        
    
    def forward(self, inputs):
        text, pos, deprel, aspect_head, aspect_mask, gather_idx, path = inputs # input features, shape (batch_size, seq_len) except for the path whose shape is (batch_size, seq_len, path_len)
        ''' common variables '''
        text_len = torch.sum(text!=0, dim=-1) # length of sentences, shape (batch_size)
        text_mask = (text!=0).unsqueeze(-1) # mask of texts (paddings: 0, others: 1), shape (batch_size, seq_len)
        aspect_mask = aspect_mask.unsqueeze(-1) # mask of aspects (aspects: 1, others: 0), shape (batch_size, seq_len)
        ''' embedding '''
        word_feature = self.embed_dropout(self.word_embedding(text)) # embed words to vectors, shape (batch_size, seq_len, word_emb_dim)
        pos_feature = self.embed_dropout(self.pos_embedding(pos)) # embed pos tags to vectors, shape (batch_size, seq_len, pos_emb_dim)
        deprel_feature = self.embed_dropout(self.deprel_embedding(deprel)) # emb dependency relations to vectors, shape (batch_size, seq_len, dep_emb_dim)
        extrel_feature = self.embed_dropout(self.ext_rel) # embedding vector of extra relation, shape (dep_emb_dim)
        ''' basic feature '''
        node_feature, _ = self.rnn(torch.cat((word_feature, pos_feature), dim=-1), text_len) # node representations, shape (batch_size, seq_len, hidden_dim*2)
        BS, SL, FD = node_feature.shape # shape of node representations
        extrel_feature = extrel_feature.reshape(1, 1, -1).expand(BS, SL, -1) # expand extra relation embedding, shape (batch_size, seq_len, dep_emb_dim)
        padword_feature = self.pad_word.reshape(1, 1, -1).expand(BS, -1, -1) # expand padding word embedding, shape (batch_size, 1, hidden_dim*2)
        exttext_feature = self.pad_word.reshape(1, 1, -1).expand(BS, SL, -1) # expand extra text feature, shape (batch_size, seq_len, hidden_dim*2)
        padedge_feature = self.pad_edge.reshape(1, 1, -1).expand(BS, -1, -1) # expand padding edge embedding, shape (batch_size, 1, 1)
        ''' arrange node '''
        gather_idx = gather_idx.unsqueeze(0).expand(FD, -1, -1) # indices for gathering the original words, shape (hidden_dim*2, batch_size, seq_len)
        node_feature = torch.cat((padword_feature, node_feature), dim=1).permute(2, 0, 1) # padded words, shape (hidden_dim*2, batch_size, seq_len)
        node_feature = torch.gather(node_feature, 2, gather_idx).permute(1, 2, 0) # original words, shape (batch_size, seq_len, hidden_dim*2)
        ''' edge feature '''
        deptext_feature = torch.cat((padword_feature, node_feature), dim=1).permute(2, 0, 1) # dependents features, shape (hidden_dim*2, batch_size, seq_len+1)
        aspect_head = aspect_head.unsqueeze(0).expand(FD, -1, -1) # head indices of current aspect, shape (hidden_dim*2, batch_size, seq_len)
        deptext_feature = torch.gather(deptext_feature, 2, aspect_head).permute(1, 2, 0) # permuted dependents features, shape (batch_size, seq_len, hidden_dim*2)
        head_text_feature = torch.cat((deptext_feature, node_feature), dim=1) # the features of start node at each edge, shape (batch_size, seq_len*2, hidden_dim*2)
        tail_text_feature = torch.cat((node_feature, exttext_feature), dim=1) # the features of end node at edge edge, shape (batch_size, seq_len*2, hidden_dim*2)
        edge_feature = torch.cat((head_text_feature, tail_text_feature), dim=-1) # the features of edges, shape (batch_size, seq_len*2, hidden_dim*4)
        ''' score function '''
        label_feature = torch.cat((deprel_feature, extrel_feature), dim=1) # compose label features, shape (batch_size, seq_len*2, dep_emb_dim)
        edge_score = torch.sigmoid(self.bilinear(self.bilinear_dropout(edge_feature), label_feature)) # compute score for each edge, shape (batch_size, seq_len*2, 1)
        edge_score = torch.cat((padedge_feature, edge_score.transpose(1, 2)), dim=-1).expand(-1, SL, -1) # expand edge scores, shape (batch_size, seq_len, seq_len*2+1)
        ''' node weight '''
        node_weight = torch.prod(torch.gather(edge_score, 2, path), dim=-1, keepdim=True) # compute node weights, shape (batch_size, seq_len, 1)
        node_weight = torch.where(text_mask!=0, node_weight, torch.zeros_like(node_weight)) # eliminate values out of texts, shape (batch_size, seq_len, 1)
        node_weight = torch.where(aspect_mask==0, node_weight, torch.zeros_like(node_weight)) # compute final node weights, shape (batch_size, seq_len, 1)
        weight_norm = torch.sum(node_weight.squeeze(-1), dim=-1) # compute L1 norm of weights, shape (batch_size)
        ''' sentence representation '''
        sentence_feature = torch.sum(node_weight * node_feature, dim=1) # compute sentence features, shape (batch_size, hidden_dim*2)
        predicts = self.fc_out(self.fc_dropout(sentence_feature)) # use fully-connected network to generate predicts, shape (batch_size, label_dim)
        return [predicts, weight_norm]
    
