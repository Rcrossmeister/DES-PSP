# DES-PSP
 Double Encoder Seq2seq-Based Presidential ConceptStock Price Prediction model

**NOTIFICATION:**

_Note : Model need interface_

*Note: The README Still need a further revise and implements.* 

__Update:__

* *A argparse interface is released. 09-015-2023*

## DataSet

The full dataset : ./data/raw/raw/All_Data.csv

## Use

* To **train** the Seq2Seq baseline model :

```shell
python main.py --model=seq2seq_lstm
```

## Results

| Model        | RMSE   | MAE    | ADE    | FDE    |
| ------------ | ------ | ------ | ------ | ------ |
| Seq2Seq_LSTM | 0.8099 | 0.5780 | 2.6031 | 0.1971 |
| Seq2Seq_GRU  | 0.8308 | 0.5881 | 2.6410 | 0.1287 |
| LSTM         | 0.8197 | 0.5792 | 2.5987 | 0.1096 |
| GRU          | 0.8570 | 0.5977 | 2.6978 | 0.2173 |
| BiLSTM       | 0.8724 | 0.6002 | 2.7056 | 0.1845 |
| BiGRU        | 0.9005 | 0.6135 | 2.7763 | 0.1528 |

## Citation

Please cite our paper if you use it in your work:

```shell
@inproceedings{,
   title={{}: },
   author={},
   booktitle={},
   year={}
}
```
