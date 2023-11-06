# DES-PSP
 Double Encoder Seq2seq-Based Presidential Conceptual Stock Price Prediction model

**NOTIFICATION:**

_Note : Model need interface._

*Note: The README Still need a further revise and implements.* 

*Note: Reproduce paper list is need to be determined.*

*Note: The test parameter need to be appointed.*

__Update:__

* *Refactored the whole repo. 10-25-2023*
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
python main_baseline.py --model=seq2seq_lstm
```



## Results

### Price

| Model       | RMSE   | MAE    | ADE    | FDE    |
| ----------- | ------ | ------ | ------ | ------ |
| BiLSTM-LSTM | 0.8385 | 0.5298 | 0.5298 | **0.6731** |
| LSTM-LSTM   | 0.8224 | **0.5242** | **0.5242** | 0.6960 |
| BiGRU-GRU   | 0.9586 | 0.6043 | 0.6043 | 0.8205 |
| GRU-GRU     | 0.8820 | 0.5509 | 0.5509 | 0.6868 |
| BiLSTM      | 0.8380 | 0.5321 | 0.5321 | 0.6690 |
| LSTM        | **0.8121** | 0.5215 | 0.5215 | 0.6862 |
| BiGRU       | 0.9215 | 0.5851 | 0.5851 | 0.7506 |
| GRU         | 0.9186 | 0.5756 | 0.5756 | 0.7729 |
| DES-PSP     | 1.0171 | 0.6373 | 0.6373 | 0.8389 |

### Movement

| Model | Accuracy | MCC |
| ----------- | ------ | ------ |
| BiLSTM-LSTM | 0.8313 | 0.0715 |
| LSTM-LSTM   | 0.8327 | 0.0197 |
| BiGRU-GRU   | **0.8084** | **0.1347** |
| GRU-GRU     | 0.8303 | 0.0497 |
| BiLSTM      | 0.8316 | 0.0392 |
| LSTM        | 0.8323 | 0.0013 |
| BiGRU       | 0.8267 | 0.0987 |
| GRU         | 0.8329 | 0.0000 |
| DES-PSP     | 0.8148 | 0.1162 |


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
