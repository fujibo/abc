import urllib.request


for i in range(1, 9):
    fc = urllib.request.urlopen(url="https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/p0{}_c.txt".format(i))
    print(int(fc.read()))
    fw = urllib.request.urlopen(url="https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/p0{}_w.txt".format(i))
    weight = []
    for element in fw:
        # print(int(element))
        weight.append(int(element))
    print(weight)

    fp = urllib.request.urlopen(url="https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/p0{}_p.txt".format(i))
    profit = []
    for element in fp:
        # print(int(element))
        profit.append(int(element))
    print(profit)

    fs = urllib.request.urlopen(url="https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/p0{}_s.txt".format(i))
    print(fs.read())
