import matplotlib.pyplot as plt
import time
import threading
import random

dataX = []
dataY = []

C = 0;

# This just simulates reading from a socket.
def data_listener():
    global C;

    while True:
        time.sleep(0.5)
        C += 1
        dataX.append( C );
        dataY.append( random.random() / 10 );

        if( len(dataX) > 20 ):
            dataX.pop(0);
            dataY.pop(0);


if __name__ == '__main__':
    # * thread = threading.Thread(target=data_listener)
    # * thread.daemon = True
    # * thread.start()
    #
    # initialize figure
    # plt.figure() 
    # ln, = plt.plot([])
    # plt.ion()
    plt.show()

    while True:
        print('pause');
        plt.pause(0.2)
        # ln.set_xdata( dataX );
        # ln.set_ydata( dataY );
        dX = dataX;
        dY = dataY;

        # plt.cla(); # clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
        plt.clf(); # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for o

        plt.plot( dX, dY );

        plt.draw();

        C += 1
        dataX.append( C );
        dataY.append( random.random() / 10 );

        if( len(dataX) > 20 ):
            dataX.pop(0);
            dataY.pop(0);
