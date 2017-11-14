import numpy as np
import random
import matplotlib.pyplot as plt


def f(x):
    """
    Energy function.
    """
    return 0.2 * np.sin(12.5 * x) + (x - 1)**2 - 5


def minimize(x, y, start_position):
    """
    Minimize the energy function.
    :param x: array, x coordinates
    :param y: array, values of the energy function
    :param start_position: int, initial position of the agent
    :return: position with the minimal found value of energy function
    """
    best_pos = start_position
    current = start_position
    T = 300
    threshold = 1
    alpha = 0.95
    path = [start_position]
    for i in range(400):
        if T < threshold: #too lazy now tomorrow is another day
            return best_pos, path
        T = alpha * T
        #decide which direction to go
        whereToGo = np.random.uniform()
        if whereToGo < 0.5:
            if current == 0: #making it a donout (comment by Homer: 'yes pleaaase!')
                next = 30
            else:
                next = current - 1
        else:
            if current == 30: #donout
                next = 0
            else:
                next = current + 1
        #decide wheather to go
        if y[next] <= y[current]: #easy goin bru
            current = next
            if y[current] < y[best_pos]:
                best_pos = current
        else: #really man?!
            effort = np.random.uniform()
            #making it 1-p to make the (tremendously) funny effort analogy
            minEffort = 1 - np.exp(-(y[next]-y[current])/T)
            if effort > minEffort: # wuhuu I got so much energy today! lets do this! Its a small step for a man but...
                current = next
                if y[current] < y[best_pos]:
                    best_pos = current
            else: # naah man its a lazy day
                #(this else block only exists to make a needless comment)
                continue
        path.append(current)
    return best_pos, path


def main():
    random.seed(2017)
    np.random.seed(2017)
    x = np.linspace(-0.5, 2, num=31, endpoint=True)
    y = f(x)
    print('x=', x)
    print('y=', y)
    start_position = 0

    best_pos, path = minimize(x, y, start_position)
    print('best value', y[best_pos], 'at position', best_pos)
    print('path:', path)

    if best_pos != np.argmin(y):
        print('You haven\'t found the global minimum. Try harder! Eat less donouts!')
    else:
        print('Success!')

    plt.plot(x, y)
    plt.plot(x, y, '--')
    plt.plot(x[start_position], y[start_position], '-bo', label='start pos', markersize=13)
    plt.plot(x[best_pos], y[best_pos], '-go', label='best found pos', markersize=13)
    plt.title('f(x)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
