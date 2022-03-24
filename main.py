import os
import traceback
from config import get_configs
from train import run_train

def main():
    P = get_configs()
    print(P, '\n')
    os.environ['CUDA_VISIBLE_DEVICES'] = P['gpu_num']
    print('###### Train start ######')
    run_train(P)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())
