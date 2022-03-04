# Intent Detection and Slot Filling

This is an extension of the work done in the paper -> `JointBERT`: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909) <br>

And the unofficial implementation by [monologg](https://github.com/monologg/JointBERT)


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
