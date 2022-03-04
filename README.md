# Intent Detection and Slot Filling

This is an extension of the work done in the paper -> `JointBERT`: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909) <br>

And the unofficial implementation by [monologg](https://github.com/monologg/JointBERT) <br>

## Objective
The objective of this project is to extend the work of Chen et al., to evaluate the performance of various Bert based model for the task of joint intent classification and slot filling and compare their performances.

### Models for Experimentation
 1. ALBERT- In 2019, Google Research researchers proposed ALBERT.The purpose of this study is to use multiple strategies such as parameter sharing, embedding matrix factorization, and inter sentence coherence loss to improve the training and outcomes of BERT architecture.

1. CANINE- It is a neural encoder that works directly on character sequences without using explicit tokenization or vocabulary, as well as a pre-training technique that works directly on characters or optionally uses subwords as a soft inductive bias.

1. ConvBERT- IT outperforms BERT and its variants in different downstream tasks, with lower training costs and fewer model parameters, according to experiments.

1. DeBERTa- (Decoding-enhanced BERT with Disentangled Attention) is a model that uses two unique strategies to improve the BERT and RoBERTa models. The first is the disentangled attention mechanism, in which each word is represented by two vectors that represent its content and position, and attention weights between words are calculated using disentangled matrices on their contents and relative positions.

1. DeBERTa –v2- The DeBERTa model has two versions. The first is DeBERTa and the second is DeBERTa v2. It comprises the 1.5B model, which scored 89.9 against the human baseline of 89.8 in the SuperGLUE single-model submission.

1. DistilBERT- It is a compact, fast, inexpensive, and light Transformer model that has been trained using BERT base. It has 40% less parameters than bert-base-uncased, and it runs 60% quicker while keeping over 95% of BERT's performance on the GLUE language understanding benchmark.

1. MPNet- To inherit the advantages of masked and permuted language modelling for natural language processing, MPNet uses a novel pre-training method called masked and permuted language modelling. 

1. RoBERTa- It extends BERT by changing crucial hyperparameters, such as deleting the next-sentence pretraining goal and training with considerably bigger mini-batches and learning rates.

## Install Requirements
```
pip install -r requirements.txt
```

## Dataset
This project is evaluated on two benchmark datasets for simultaneous Intent Detection and Slot Filling
- [The ATIS (Airline Travel Information System) Dataset](https://github.com/howl-anderson/ATIS_dataset/blob/master/README.en-US.md)
- [The SNIPS Dataset](https://github.com/sonos/nlu-benchmark) 


## Training & Evaluation

```bash
$ python3 main.py --task {task_name} \
                  --model_type {model_type} \
                  --model_dir {model_dir_name} \
                  --do_train --do_eval \
                  --use_crf

# For ATIS
$ python3 main.py --task atis \
                  --model_type bert \
                  --model_dir atis_model \
                  --do_train --do_eval
# For Snips
$ python3 main.py --task snips \
                  --model_type bert \
                  --model_dir snips_model \
                  --do_train --do_eval
```

## Prediction

```bash
$ python3 predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```


## References

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)
