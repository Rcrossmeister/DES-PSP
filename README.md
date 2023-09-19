# DES-PSP
 Double Encoder Seq2seq-Based Presidential Conceptual Stock Price Prediction model

**NOTIFICATION:**

_Note : Model need interface._

*Note: The README Still need a further revise and implements.* 

*Note: Reproduce paper list is need to be determined.*

*Note: The test parameter need to be appointed.*

__Update:__

* *A argparse interface is released. 09-015-2023*

**TO-DO:**

1. Ablation Study.
2. Need to compare the proposed DES-PSP model with existing state-of-the-art methods in the field of stock market prediction.
3. Need more comprehensive analysis of the model's sensitivity to different parameter settings.
4. Potentially combine novelty methods (After well-done baseline).
5. Verifying the hypothesis of the exists stock which benefits from the election of a specific presidential candidate.
6. Find background news to support the election of conceptual stocks.
7. State the generalizability.

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
