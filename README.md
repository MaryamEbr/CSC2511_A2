# Neural-Machine-Translation-French-English

This repository contains codes for a computer assignment for Natural Language Computing course (CSC401/2511, Winter 2022, UofT)

The project is about Neural Machine Translation (English to French). We implemented a simple seq2seq model, without attention, with single-headed attention, and
with multi-headed attention. Then trained the models with teacher forcing and decoded them using beam search. The quality of translation results is evaluated with BLEU Score.
The codes are in Python.

The main corpus for this assignment comes from the social records (Hansards) of the 36th Canadian
Parliament, including debates from both the House of Representatives and the Senate. It contains both English and French sentences.
