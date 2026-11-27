# Six-task evaluation: current accepted results

Each model/view contains 180 task instances. Delete Below contains 10 Including and 20 Excluding instances. Counts are assigned to the recorded first accepted stage and verified against final submissions. Delete Below uses online lattice-noop scoring; other task entries retain their adopted final-verification results.

## 1. Seven-count table

| Model | view | Delete Below | Delete Around | Insert Between | Move Towards | Rotate Around | Remove Molecule | Total180 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-Plus | ON | 27/1/1/0/0/0/1 | 18/4/4/0/0/1/3 | 12/9/1/0/0/0/8 | 25/3/0/0/1/0/1 | 6/2/0/2/0/0/20 | 13/2/0/0/2/0/13 | 101/21/6/2/3/1/46 |
| Qwen3-VL-Plus | OFF | 28/2/0/0/0/0/0 | 11/10/6/1/0/1/1 | 19/6/3/0/0/0/2 | 25/5/0/0/0/0/0 | 6/2/2/0/0/0/20 | 13/2/0/1/1/0/13 | 102/27/11/2/1/1/36 |
| Qwen3.5-Plus | ON | 24/6/0/0/0/0/0 | 16/6/4/3/0/0/1 | 20/4/1/0/1/0/4 | 23/2/3/1/0/0/1 | 12/0/0/1/3/2/12 | 16/4/3/1/2/1/3 | 111/22/11/6/6/3/21 |
| Qwen3.5-Plus | OFF | 25/4/1/0/0/0/0 | 22/4/3/1/0/0/0 | 24/0/0/2/0/0/4 | 27/2/1/0/0/0/0 | 11/3/2/0/2/1/11 | 17/5/3/2/2/0/1 | 126/18/10/5/4/1/16 |
| Qwen3.7-Plus | ON | 29/1/0/0/0/0/0 | 19/5/4/0/0/1/1 | 24/5/0/0/1/0/0 | 24/5/1/0/0/0/0 | 9/6/0/2/1/0/12 | 14/12/2/2/0/0/0 | 119/34/7/4/2/1/13 |
| Qwen3.7-Plus | OFF | 27/3/0/0/0/0/0 | 22/7/1/0/0/0/0 | 23/2/1/1/0/2/1 | 30/0/0/0/0/0/0 | 9/3/4/1/1/1/11 | 18/9/0/1/1/0/1 | 129/24/6/3/2/3/13 |
| Qwen3.8-Max | ON | 30/0/0/0/0/0/0 | 26/3/1/0/0/0/0 | 16/6/1/2/1/1/3 | 25/5/0/0/0/0/0 | 11/3/2/2/1/0/11 | 18/9/2/1/0/0/0 | 126/26/6/5/2/1/14 |
| Qwen3.8-Max | OFF | 26/4/0/0/0/0/0 | 23/7/0/0/0/0/0 | 16/5/4/1/0/1/3 | 19/9/1/1/0/0/0 | 4/9/3/0/2/1/11 | 20/5/0/1/1/2/1 | 108/39/8/3/3/4/15 |
| DeepSeek-V4-Flash | ON | 29/1/0/0/0/0/0 | 23/6/1/0/0/0/0 | 19/6/3/0/1/1/0 | 26/3/0/0/0/0/1 | 12/5/4/3/3/0/3 | 13/7/4/1/0/1/4 | 122/28/12/4/4/2/8 |
| DeepSeek-V4-Flash | OFF | 24/5/1/0/0/0/0 | 15/11/3/0/0/1/0 | 16/6/5/1/0/0/2 | 24/4/2/0/0/0/0 | 12/7/0/3/0/0/8 | 20/7/1/1/0/0/1 | 111/40/12/5/0/1/11 |

## 2. Task/view totals

| Task | view | initial | repair1 | repair2 | repair3 | repair4 | repair5 | fail |
|---|---|---|---|---|---|---|---|---|
| Delete Below | ON | 139 | 9 | 1 | 0 | 0 | 0 | 1 |
| Delete Below | OFF | 130 | 18 | 2 | 0 | 0 | 0 | 0 |
| Delete Below | ON+OFF | 269 | 27 | 3 | 0 | 0 | 0 | 1 |
| Delete Around | ON | 102 | 24 | 14 | 3 | 0 | 2 | 5 |
| Delete Around | OFF | 93 | 39 | 13 | 2 | 0 | 2 | 1 |
| Delete Around | ON+OFF | 195 | 63 | 27 | 5 | 0 | 4 | 6 |
| Insert Between | ON | 91 | 30 | 6 | 2 | 4 | 2 | 15 |
| Insert Between | OFF | 98 | 19 | 13 | 5 | 0 | 3 | 12 |
| Insert Between | ON+OFF | 189 | 49 | 19 | 7 | 4 | 5 | 27 |
| Move Towards | ON | 123 | 18 | 4 | 1 | 1 | 0 | 3 |
| Move Towards | OFF | 125 | 20 | 4 | 1 | 0 | 0 | 0 |
| Move Towards | ON+OFF | 248 | 38 | 8 | 2 | 1 | 0 | 3 |
| Rotate Around | ON | 50 | 16 | 6 | 10 | 8 | 2 | 58 |
| Rotate Around | OFF | 42 | 24 | 11 | 4 | 5 | 3 | 61 |
| Rotate Around | ON+OFF | 92 | 40 | 17 | 14 | 13 | 5 | 119 |
| Remove Molecule | ON | 74 | 34 | 11 | 5 | 4 | 2 | 20 |
| Remove Molecule | OFF | 88 | 28 | 4 | 6 | 5 | 2 | 17 |
| Remove Molecule | ON+OFF | 162 | 62 | 15 | 11 | 9 | 4 | 37 |

## 3. Three-count table

| Model | view | Delete Below | Delete Around | Insert Between | Move Towards | Rotate Around | Remove Molecule | Total180 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-Plus | ON | 27/2/1 | 18/9/3 | 12/10/8 | 25/4/1 | 6/4/20 | 13/4/13 | 101/33/46 |
| Qwen3-VL-Plus | OFF | 28/2/0 | 11/18/1 | 19/9/2 | 25/5/0 | 6/4/20 | 13/4/13 | 102/42/36 |
| Qwen3.5-Plus | ON | 24/6/0 | 16/13/1 | 20/6/4 | 23/6/1 | 12/6/12 | 16/11/3 | 111/48/21 |
| Qwen3.5-Plus | OFF | 25/5/0 | 22/8/0 | 24/2/4 | 27/3/0 | 11/8/11 | 17/12/1 | 126/38/16 |
| Qwen3.7-Plus | ON | 29/1/0 | 19/10/1 | 24/6/0 | 24/6/0 | 9/9/12 | 14/16/0 | 119/48/13 |
| Qwen3.7-Plus | OFF | 27/3/0 | 22/8/0 | 23/6/1 | 30/0/0 | 9/10/11 | 18/11/1 | 129/38/13 |
| Qwen3.8-Max | ON | 30/0/0 | 26/4/0 | 16/11/3 | 25/5/0 | 11/8/11 | 18/12/0 | 126/40/14 |
| Qwen3.8-Max | OFF | 26/4/0 | 23/7/0 | 16/11/3 | 19/11/0 | 4/15/11 | 20/9/1 | 108/57/15 |
| DeepSeek-V4-Flash | ON | 29/1/0 | 23/7/0 | 19/11/0 | 26/3/1 | 12/15/3 | 13/13/4 | 122/50/8 |
| DeepSeek-V4-Flash | OFF | 24/6/0 | 15/15/0 | 16/12/2 | 24/6/0 | 12/10/8 | 20/9/1 | 111/58/11 |
