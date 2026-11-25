# Dataset Split and Spec Files

## GOT-10k
- `got10k_train_full_split.txt`: all 9,335 official GOT-10k training sequence indices (`0..9334`).
- `got10k_train_split.txt`: training subset of the GOT-10k training set.
- `got10k_val_split.txt`: validation subset of the GOT-10k training set.
- `got10k_vot_exclude.txt`: 1,000 videos excluded from training when evaluating on VOT (as required by [VOT Challenge](https://www.votchallenge.net/vot2020/participation.html)).
- `got10k_vot_train_split.txt`: training subset of the VOT-permitted GOT-10k training set.
- `got10k_vot_val_split.txt`: validation subset of the VOT-permitted GOT-10k training set.

## LaSOT
- `lasot_train_split.txt`: complete LaSOT training split.

## TrackingNet
- `trackingnet_classmap.txt`: mapping from sequence name to target object class for TrackingNet.

## VastTrack
- `vasttrack_train_split.txt`: VastTrack training sequence list (`<class>/<sequence>`).
- `vasttrack_test_list.txt`: VastTrack test sequence list (`<class>/<sequence>`).
- `vasttrack_broken_split.txt`: optional exclusion list for corrupted sequences (empty by default).
