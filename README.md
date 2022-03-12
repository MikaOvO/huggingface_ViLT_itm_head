# huggingface_ViLT_itm_head

huggingface上ViLT没有itm这个预训练的head，但是有一个对应的预训练后的checkpoint（链接：https://huggingface.co/dandelin/vilt-b32-mlm-itm）

这里简单写一下模型head结构，以便使用，实测加载后没有问题

将bin文件下载以后
```python
model = ViltForPreTrain('pytorch.bin')
```

就可以使用啦，在forward时需要指明使用哪个head，inputs和labels参考官方文档，对于itm的labels需要给出(batch_size, 1)的tensor指明类别
huggingface ViLT(mlm) doc: https://huggingface.co/docs/transformers/master/en/model_doc/vilt#transformers.ViltForMaskedLM
