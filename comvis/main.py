# run the program
import argparse
import backbone.main_net as net
import train as train
import test as test

#define argument
parser = argparse.ArgumentParser()
parser.add_argument('--command', type=str, required=True, help='train for training, test for testing, net for print layer used, hello for testing')

args = parser.parse_args()

if __name__ == '__main__':
    if args.command == 'train':
        print('trian.py')
    if args.command == 'test':
        print('test')
    if args.command == 'net':
        print(net.MainNet())
    if args.command == 'hello':
        print('hello!')