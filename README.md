# bert4ner.pytorch
A [PyTorch](https://pytorch.org/) implementation of [Bert](https://arxiv.org/abs/1706.03762) models for Named Entity Recognition(NER).

### Quick start

The training arguments can be found in `args/`. To train CRF models, the command should be like
``` sh
$ python run_ner_crf.py args/bert_crf-clue_ner.json
```

### Requirements

RTX3090, Driver Version: 455.38, CUDA Version: 11.1
- apex==0.1
- numpy==1.19.4
- pandas==1.1.4
- scipy==1.5.4
- seqeval==1.2.2
- torch==1.7.0
- torchtyping==0.1.1
- tqdm==4.49.0
- transformers==4.5.0

### Datasets

Datasets used in this repository can be found in [OYE93/Chinese-NLP-Corpus - Github](https://github.com/OYE93/Chinese-NLP-Corpus) and [CLUEbenchmark](https://github.com/CLUEbenchmark/).

- [msra_ner](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/MSRA)
- [peoples_daily_ner](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/People's%20Daily)
- [weibo_ner](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/Weibo)
- [cluener2000](https://www.cluebenchmarks.com/dataSet_search_modify.html?keywords=cluener2000)

### Models

Pretrained language model(PLM) including
- [BERT](https://arxiv.org/abs/1706.03762)
- [NEZHA](https://arxiv.org/abs/1909.00204)
- [FLAT](https://arxiv.org/abs/2004.11795)

Named Entity Recognition Prediction Header including
- [CRF](https://arxiv.org/abs/1508.01991): Bidirectional LSTM-CRF Models for Sequence Tagging
- [SPAN]()
- [Biaffine](https://arxiv.org/abs/2005.07150): Named Entity Recognition as Dependency Parsing
<!-- - [MRC-modified](https://arxiv.org/abs/1910.11476): A Unified MRC Framework for Named Entity Recognition -->

### Benchmarks

#### Models

The performance on **dev** set of `CLUENER2000`(F1 score).
| entitiy       | Human Performance | Bert-crf | Bert-span | Bert-biaffine |
|:-------------:|:-----:|:-----:|:-----:|:-----:|
| Address       | 43.04 | 64.30 | 65.76 | 65.67 |
| Book          | 71.70 | 82.58 | 83.61 | 83.39 |
| Company       | 49.32 | 80.82 | 83.11 | 81.75 |
| Game          | 80.39 | 84.93 | 85.29 | 83.52 |
| Government    | 79.27 | 82.92 | 83.07 | 80.23 |
| Movie         | 63.21 | 83.06 | 83.11 | 82.19 |
| Person Name   | 74.49 | 88.87 | 88.87 | 87.83 |
| Organization  | 65.41 | 80.27 | 82.48 | 81.05 |
| Position      | 55.38 | 79.51 | 80.69 | 80.14 |
| Scene         | 51.85 | 76.56 | 72.37 | 73.17 |
| Overall@Macro | 63.41 | 80.38 | 80.84 | 79.89 |
| Overall@Micro | /     | 80.26 | 81.00 | 80.04 |

#### Datasets

### Reference

- [huggingface/datasets - Github](https://github.com/huggingface/datasets)
- [OYE93/Chinese-NLP-Corpus - Github](https://github.com/OYE93/Chinese-NLP-Corpus)
- [lonePatient/BERT-NER-Pytorch - Github](https://github.com/lonePatient/BERT-NER-Pytorch)
- [LeeSureman/Flat-Lattice-Transformer - Github](https://github.com/LeeSureman/Flat-Lattice-Transformer)
- [ShannonAI/mrc-for-flat-nested-ner - Github](https://github.com/ShannonAI/mrc-for-flat-nested-ner)