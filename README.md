# Homework 1 ADL NTU

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip install -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection
```shell
# training
python train_intent.py \
    --max_len 128 \
    --model_type "GRU" \
    --dropout 0.3 \
    --lr 1e-3 \
    --L2 3e-4 \
    --batch_size 512 \
    --device "cuda" \
    --num_epoch 50
```

```shell
# testing if gpu is available
bash intent_cls.sh /path/to/test.json /path/to/pred.csv
```

## Slot tagging
```shell
# training
python train_slot.py \
    --max_len 128 \
    --model_type "LSTM" \
    --L2 1e-4 \
    --device "cuda" \
    --num_epoch 50
```

```shell
# testing if gpu is available
bash slot_tag.sh /path/to/test.json /path/to/pred.csv
```