"""
Created on Wed Jun 09 11:04:36 2020

@author: Aayushi Agarwal
"""
import cv2
from speedCheck import SpeedCheck
import argparse

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    speed = SpeedCheck(config)
    hf_factor = speed.run()
    speed.test(hf_factor, config.save_test_txt)
    cv2.destroyAllWindows()



if __name__=='__main__':
    parser = argparse.ArgumentParser()

    
    parser.add_argument('--trainvid', type=str, default='data/train.mp4')
    parser.add_argument('--testvid', type=str, default='data/test.mp4')
    parser.add_argument('--traintxt', type=str, default='data/train.txt')
    parser.add_argument('--visual', type=str2bool, default=True)
    parser.add_argument('--gt_len', type=int, default=20400)
    parser.add_argument('--save_test_txt', type=str2bool, default=True)
    config = parser.parse_args()

    main(config)

