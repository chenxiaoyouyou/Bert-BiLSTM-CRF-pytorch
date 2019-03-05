# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from config import Config
from model import BERT_LSTM_CRF
import torch.optim as optim
from utils import load_vocab, read_corpus, load_model, save_model
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import fire


def train(**kwargs):
    config = Config()
    config.update(**kwargs)
    print('当前设置为:\n', config)
    if config.use_cuda:
        torch.cuda.set_device(config.gpu)
    print('loading corpus')
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.label_file)
    tagset_size = len(label_dic)
    train_data = read_corpus(config.train_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    dev_data = read_corpus(config.dev_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)

    train_ids = torch.LongTensor([temp.input_id for temp in train_data])
    train_masks = torch.LongTensor([temp.input_mask for temp in train_data])
    train_tags = torch.LongTensor([temp.label_id for temp in train_data])

    train_dataset = TensorDataset(train_ids, train_masks, train_tags)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)

    dev_ids = torch.LongTensor([temp.input_id for temp in dev_data])
    dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data])
    dev_tags = torch.LongTensor([temp.label_id for temp in dev_data])

    dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=config.batch_size)
    model = BERT_LSTM_CRF(config.bert_path, tagset_size, config.bert_embedding, config.rnn_hidden, config.rnn_layer, dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda)
    if config.load_model:
        assert config.load_path is not None
        model = load_model(model, name=config.load_path)
    if config.use_cuda:
        model.cuda()
    model.train()
    optimizer = getattr(optim, config.optim)
    optimizer = optimizer(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    eval_loss = 10000
    for epoch in range(config.base_epoch):
        step = 0
        for i, batch in enumerate(train_loader):
            step += 1
            model.zero_grad()
            inputs, masks, tags = batch
            inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
            if config.use_cuda:
                inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()
            feats = model(inputs, masks)
            loss = model.loss(feats, masks,tags)
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print('step: {} |  epoch: {}|  loss: {}'.format(step, epoch, loss.item()))
        loss_temp = dev(model, dev_loader, epoch, config)
        if loss_temp < eval_loss:
            save_model(model,epoch)


def dev(model, dev_loader, epoch, config):
    model.eval()
    eval_loss = 0
    true = []
    pred = []
    length = 0
    for i, batch in enumerate(dev_loader):
        inputs, masks, tags = batch
        length += inputs.size(0)
        inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
        if config.use_cuda:
            inputs, masks, tags = inputs.cuda(), masks.cuda(), tags.cuda()
        feats = model(inputs, masks)
        path_score, best_path = model.crf(feats, masks.byte())
        loss = model.loss(feats, masks, tags)
        eval_loss += loss.item()
        pred.extend([t for t in best_path])
        true.extend([t for t in tags])
    print('eval  epoch: {}|  loss: {}'.format(epoch, eval_loss/length))
    model.train()
    return eval_loss


if __name__ == '__main__':
    fire.Fire()










