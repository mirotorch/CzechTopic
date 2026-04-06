# CzechTopic

Different encoding approaches:
- cross-encoder (like in the original paper)
- bi-encoder (ColBERT-like architecture looks promising)
- two-level retrieval-localization arch combining both (really? source is not reliable)

Matrix can be replaced with another head and or expanded with additional layer like MLP, CRF, LSTM/GRU.

We can also use 2 different heads, one for theme detection and second for localization, using arbitrary encoding style.

It's also possible to experiment with training objective, e.g. 

Finally, we can try to change the represenation of the topic, like using only CLS, or using only description.

Cross-encoder + better token head
Multitask text-level + token-level training
ColBERT-like late interaction
Two-stage retrieval/localization
