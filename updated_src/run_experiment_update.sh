#!/bin/bash
echo "Yelp training"

# python run_multi_seed.py --data_name yelp --n_seeds 10 --losses contrastive --cuda_id 0

python run_multi_seed.py --data_name yelp --n_seeds 10 --losses margin --cuda_id 0

# python run_multi_seed.py --data_name yelp --n_seeds 10 --losses dice --cuda_id 0

# python run_multi_seed.py --data_name yelp --n_seeds 10 --losses huber --cuda_id 2
