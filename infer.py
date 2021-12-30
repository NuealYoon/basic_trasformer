import math
import torch
# import torch.nn as nn
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout( p = dropout )

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# 1. Define the model
class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

# 2. Load and batch data
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    # 3. Initiate an instance
    ntokens = len(vocab)  # size of vocabulary
    emsize = 200  # embedding dimension
    d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

    # 저장된 weight 불러 오기
    model_weight = torch.load('model')
    model.load_state_dict(model_weight['model_state_dict'])

    print('best val loss: {}'.format(model_weight['best_val_loss']))
    print('test loss: {}'.format(model_weight['test loss']))
    print('ppl: {}'.format(model_weight['test ppl']))

    # 데이터 확인
    if False:
        train_iter, val_iter, test_iter = WikiText2()
        for index, item in enumerate(train_iter):
            # print(index)
            print(item)
            tokens = tokenizer(item)
            ids = vocab(tokens)
            print(tokens)
            print(ids)
            ids_tensor = torch.tensor(ids)
            b = 0


    # infer
    model.eval()

    '''
    학습 데이터 예제
     = = = = West Indies = = = = 
     The common starling was introduced to Jamaica in 1903 , and the Bahamas and Cuba were colonised naturally from the US . This bird is fairly common but local in Jamaica , Grand <unk> and <unk> , and is rare in the rest of the Bahamas , eastern Cuba , the Cayman Islands , Puerto Rico and St. Croix . 
    '''
    text = 'The common starling was introduced to Jamaica in 1903 ,'
    tokens = tokenizer(text)
    ids = torch.tensor(vocab(tokens))

    bptt = 35 # bptt(Backpropagation Through Time): 입력되는 모든 시퀀스 데이터에 대해서 학습 되는 시간, 보통 k개 만큼만 학습하는데, 이를 truncated-BPTT라고 한다.
    # src_mask = generate_square_subsequent_mask(bptt).to(device)
    src_mask = generate_square_subsequent_mask(bptt)
    src_mask = src_mask[:ids.size(0), :ids.size(0)]

    ids         = ids.to(device)
    src_mask    = src_mask.to(device)

    ids = ids.unsqueeze(1)
    output = model(ids, src_mask)


    sentence = ''
    for data in output:
        # print(data)
        # torch.argmax(data[0])
        word = vocab.vocab.itos_[torch.argmax(data[0]).item()]
        sentence = sentence + ' ' + word

        bb = 0
    print('입력: {}'.format(text))
    print('결과: {}'.format(sentence))



    
    #
    # print(output.shape)
    # torch.argmax(output[0][0])
