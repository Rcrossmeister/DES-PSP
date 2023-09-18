# DES-PSP
 Double Encoder Seq2seq-Based Presidential ConceptStock Price Prediction model

## Updates

- 2023.09.14
  - modify with `argparse`
  - `results/`

## Results

| Model        | RMSE   | MAE    | ADE    | FDE    |
| ------------ | ------ | ------ | ------ | ------ |
| Seq2Seq_LSTM | 0.8099 | 0.5780 | 2.6031 | 0.1971 |
| Seq2Seq_GRU  | 0.8308 | 0.5881 | 2.6410 | 0.1287 |
| LSTM         | 0.8197 | 0.5792 | 2.5987 | 0.1096 |
| GRU          | 0.8570 | 0.5977 | 2.6978 | 0.2173 |
| BiLSTM       | 0.8724 | 0.6002 | 2.7056 | 0.1845 |
| BiGRU        | 0.9005 | 0.6135 | 2.7763 | 0.1528 |

