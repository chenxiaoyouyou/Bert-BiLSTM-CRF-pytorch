# Bert-BiLSTM-CRF-pytorch
使用谷歌预训练bert做字嵌入的BiLSTM-CRF序列标注模型

本模型使用谷歌预训练bert模型（https://github.com/google-research/bert）， 
同时使用pytorch-pretrained-BERT（https://github.com/huggingface/pytorch-pretrained-BERT）
项目加载bert模型并转化为pytorch参数，CRF代码参考了SLTK（https://github.com/liu-nlper/SLTK）

准备数据格式参见data

模型参数可以在config中进行设置

运行代码

python main.py train --use_cuda=False --batch_size=10  

pytorch.bin  百度网盘链接   链接:https://pan.baidu.com/s/160cvZXyR_qdAv801bDY2mQ 提取码:q67r 
