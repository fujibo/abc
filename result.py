import numpy as np
import matplotlib.pyplot as plt

def main(num):
    f = open("log.txt", 'r')
    paramArr = (
        "param: maxiter=50, bees=(50, 50, 5), H=5",
        "param: maxiter=50, bees=(30, 30, 3), H=3",
        "param: maxiter=50, bees=(10, 10, 1), H=1",
        "param: maxiter=50, bees=(3, 3, 1), H=1",
        "param: p=(4e-2, 12e-2)",
        "param: p=(1e-2, 3e-2)",
        "param: p=(0, 0)"
    )

    ansArr = [[] for i in range(7)]
    aveArr = [None for i in range(7)]
    while True:
        line = f.readline()
        if line == "":
            break
        if line == "problem" + str(num) + "\n":
            for j in range(7):
                for i in range(4):
                    tmp = f.readline()
                    tmp = tmp[0:-1]
                    if i == 0:
                        param = tmp
                        idx = paramArr.index(param)
                    elif i == 1:
                        bin_ans = list(map(int, tmp[1:-1].split(" ")))
                    elif i == 2:
                        ans = list(map(int, tmp[1:-1].split(", ")))
                        ansArr[idx].append(ans[-1])
                    else:
                        ave = list(map(float, tmp[1:-1].split(", ")))
                else:
                #     print(param)
                #
                    print(bin_ans)
                    print(ans[-1])
                    # print(ave)

            else:
                f.readline()[0:-1]
                BIN_ANS = f.readline()[0:-1]
                ANS = int(f.readline())
                # print(ANS)
        else:
            continue

    print(BIN_ANS)
    # print(paramArr)
    ansArr = np.array(ansArr)
    # print(np.mean(ansArr, axis=1))
    # print(np.std(ansArr, axis=1, ddof=1))
    return(paramArr, ansArr, ANS)


if __name__ == '__main__':
    p, a, ans = main(3)
    # print(p)
    # print(a)
    for i in range(7):
        print(p[i])
        print(a[i])
    print(ans)
    error = np.std(a, axis=1, ddof=1) / np.sqrt(a.shape[1])
    plt.bar(np.arange(1, 8), np.mean(a, axis=1), color="green", align="center", yerr=error, ecolor="black")
    plt.hlines(ans, 0, 8, colors='r')
    plt.xticks(np.arange(1, 8))
    plt.xlabel('number')
    plt.ylabel('profit')
    # plt.xlabel(p)
    plt.show()
