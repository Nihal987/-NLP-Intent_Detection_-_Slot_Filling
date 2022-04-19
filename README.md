# Intent Detection and Slot Filling

This is an extension of the work done in the paper -> `JointBERT`: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909) <br>

And the unofficial implementation by [monologg](https://github.com/monologg/JointBERT) <br>

## Objective
The objective of this project is to extend the work of Chen et al., to evaluate the performance of various Bert based model for the task of joint intent classification and slot filling and compare their performances.

### Models for Experimentation
 #### Models that worked
 1. Albert
 2. Distilbert
 3. Bertweet
 4. MPNet
 5. Roberta
 6. MobileBERT
 7. XLNET
 8. Squeezbert
 9. BERTweet
 #### Models that failed
 1. CANINE
 2. BART
 3. BigBird
 4. ConvBERT
 5. DeBerta
 6. DeBERTa –v2
 7. MBart
 8. OpenaiGPT2

## Results
| Models            | Intent acc (%) | Slot F1 (%) | Sentence acc (%) |
| ----------------- | -------------- | ----------- | ---------------- |
| BERT              | 97.74          | 95.99       | 88.24            |
| BERT + CRF        | 97.42          | 95.83       | 88.02            |
| DistilBERT        | 97.54          | 95.28       | 86.9             |
| DistilBERT + CRF  | 97.42          | 95.89       | 88.24            |
| ALBERT            | 97.64          | 95.78       | 88.13            |
| ALBERT + CRF      | 97.42          | 96.32       | 88.69            |
| RoBERTa           | 97.87          | 95.27       | 87.23            |
| RoBERTa + CRF     | 97.42          | 95.81       | 88.02            |
| MPNET             | 92.61          | 84.89       | 63.83            |
| MPNET + CRF       | 87.34          | 95          | 98.95            |
| BERTWEET          | 1.9            | 0.23        | 0                |
| BERTWEET + CRF    | 0.78           | 0.27        | 0                |
| MobileBERT        | 97.76          | 95.13       | 86               |
| MobileBERT + CRF  | 97.36          | 95.17       | 86.56            |
| SqueezeBERT       | 95.63          | 80.85       | 55.65            |
| SqueezeBERT + CRF | 89.7           | 88.42       | 67.2             |
| XLNET             | 97.31          | 94.1        | 83.54            |
| XLNET + CRF       | 97.1           | 94.91       | 84.99            |

## Install Requirements
```
pip install -r requirements.txt
```

## Dataset
This project is evaluated on the ATIS benchmark dataset for simultaneous Intent Detection and Slot Filling
- [The ATIS (Airline Travel Information System) Dataset](https://github.com/howl-anderson/ATIS_dataset/blob/master/README.en-US.md)


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

## Sample Jupyter Notebook
I have created a [sample notebook](Intent%2BSlot_filling.ipynb) that can be run on Google Colab 


## References

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)
