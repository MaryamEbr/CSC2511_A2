python3.9 a2_run.py train $TRAIN \
vocab_tiny.e.gz vocab_tiny.f.gz \
train_tiny.txt.gz dev_tiny.txt.gz \
model.pt.gz \
--epochs 2 \
--word-embedding-size 51 \
--encoder-hidden-size 100 \
--batch-size 5 \
--cell-type gru \
--beam-width 2
--with-attention




python3.9 a2_run.py train $TRAIN \
vocab.e.gz vocab.f.gz \
train.txt.gz dev.txt.gz \
model_wo_att.pt.gz \