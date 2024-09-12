import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp, paddle, newsData

import oobe

from functools import partial
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import MapDataset

from __utils import convert_example, read_vocab, write_results

# Process
USEGPU = oobe.UseGPU()

# load()
train_ds = newsData.NewsData(oobe.FileConv(r'readonly\train.txt'), mode = 'train')
dev_ds = newsData.NewsData(oobe.FileConv(r'readonly\dev.txt'), mode = 'dev')
test_ds = newsData.NewsData(oobe.FileConv(r'readonly\test.txt'), mode = 'test')

def create_dataloader(
    dataset: paddle.io.Dataset,
    trans_fn = None,
    mode = 'train',
    batch_size = 1,
    use_gpu = False,
    batchify_fn = None
) -> paddle.io.DataLoader:
    if trans_fn:
        dataset = MapDataset(dataset)
        dataset = dataset.map(trans_fn)

    if mode == 'train' and use_gpu:
        sampler = paddle.io.DistributedBatchSampler(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = True
        )
    
    else:
        shuffle = True if mode == 'train' else False
        sampler = paddle.io.BatchSampler(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle
        )
    
    dataloader = paddle.io.DataLoader(
        dataset = dataset,
        batch_sampler = sampler,
        collate_fn = batchify_fn
    )

    return dataloader

vocab = read_vocab(oobe.FileConv(r'train\vocab.txt'))
stopwords = read_vocab(oobe.FileConv(r'readonly\stopwords_baidu.txt'))
batchsize = 128
epochs = 2
trans_fn = partial(convert_example, vocab = vocab, stop_words = stopwords, is_test = False)

batchify_fn = lambda samples, fn = Tuple(
    Pad(axis = 0, pad_val = vocab.get('[PAD]', 0)),
    Stack(dtype = 'int64'),
    Stack(dtype = 'int64')
): [data for data in fn(samples)]

train_loader = create_dataloader(
    train_ds,
    trans_fn = trans_fn,
    batch_size = batchsize,
    use_gpu = USEGPU,
    batchify_fn = batchify_fn
)

dev_loader = create_dataloader(
    dev_ds,
    trans_fn = trans_fn,
    batch_size = batchsize,
    mode = 'validation', # mode = 'dev'???
    use_gpu = USEGPU,
    batchify_fn = batchify_fn
)

# Train

class LSTMModel(nn.Layer):
    def __init__(self, vocab_size, num_classes, emb_dim = 128, padding_idx = 0, lstm_hidden_size = 198, direction = 'forward', 
                 lstm_layers = 1, dropout_rate = 0.0, pooling_type = None, fc_hidden_size = 96):
        super().__init__()

        # word_id -> word_embedding
        self.embedder = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = emb_dim,
            padding_idx = padding_idx,
        )

        # word_embedding -(LSTMEncoder)-> sentence_embedding
        self.lstm_encoder = paddlenlp.seq2vec.LSTMEncoder(
            emb_dim,
            lstm_hidden_size,
            num_layers = lstm_layers,
            # ?
            direction = direction,
            dropout = dropout_rate,
            pooling_type = pooling_type
        )

        self.fc = nn.Linear(self.lstm_encoder.get_output_dim(), fc_hidden_size)
        self.output_layer = nn.Linear(fc_hidden_size, num_classes)

    def forward(self, text, seq_len):
        embedded_text = self.embedder(text)

        text_repr = self.lstm_encoder(embedded_text, sequence_length = seq_len)

        fc_out = paddle.tanh(self.fc(text_repr))
        logits = self.output_layer(fc_out)

        return logits

model = LSTMModel(
    len(vocab),
    len(train_ds.label_list),
    direction = 'bidirectional',
    padding_idx = vocab['[PAD]']
)

model = paddle.Model(model)

# Real Train

optimizer = paddle.optimizer.Adam(parameters = model.parameters(), learning_rate = 5e-4)
criterion = paddle.nn.CrossEntropyLoss()
metric = paddle.metric.Accuracy()

model.prepare(optimizer, criterion, metric)
model.fit(train_loader, dev_loader, epochs = epochs, save_dir = oobe.FileConv('log'))