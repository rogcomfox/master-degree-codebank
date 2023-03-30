# run the program
import argparse
import backbone.main_net as net
import train as train
import test as test

#define usage msg
def msg(name=None):
    return '''main.py --command
            [train, for training]
            [test, for testing]
            [net, show cnn layer]
            [hello, for view something]
            comment
            more comment
        '''

#define argument
parser = argparse.ArgumentParser(usage=msg())
parser.add_argument('--command', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    if args.command == 'train':
        #TODO: for testing purpose
        train.train("")
    elif args.command == 'test':
        print('test')
    elif args.command == 'net':
        print(net.MainNet())
    elif args.command == 'hello':
        print('This Code is implemented from paper Titled "Vehicle classification using a real-time convolutional structure based on DWT pooling layer and SE blocks"')