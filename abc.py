import numpy as np
import itertools
import time

class ABC(object):
    """docstring for ABC
    solving KP by using ABC algorithm
    """
    def __init__(self, size=10):
        'set datasize, data(problem) and set parameters for ABC'
        self.size = size
        self.getdata()
        self.set_params()

    def getdata(self):
        'set a problem'
        self.profit = np.random.randint(1, 30, self.size)
        self.weight = np.random.randint(1, 30, self.size)
        self.C = np.random.randint(1 * self.size, 10 * self.size)

    def set_params(self, maxiter=50, bees=(50, 50, 5), p=(4e-2, 12e-2), mu=3, beta=-10, gamma=16, H=5):
        'set parameters'
        self.maxIter = maxiter
        self.employed = bees[0]
        self.onlooker = bees[1]
        self.scout = bees[2]
        self.p_min = p[0]
        self.p_max = p[1]
        self.mu = mu
        self.beta = beta
        self.gamma = gamma
        self.H = H

    def initAns(self):
        '''return initial answer for this KP
        return matrix(the number of bees x self.size)
        '''
        self.relation = self.profit / self.weight
        T = np.sum(self.weight)
        prob = self.C / T * self.relation / np.mean(self.relation)

        def modify():
            'in x, discard the worse items'
            tmp = self.relation * x
            np.place(tmp, tmp==0.0, np.nan)
            try:
                idx = np.nanargmin(tmp)
            except Exception as e:
                raise e
            x[idx] = False
            return self.weight.dot(x)

        xs = []
        for i in range(self.employed + self.scout):
            x = np.random.rand(self.size) < self.relation
            W = self.weight.dot(x)

            while self.C < W:
                try:
                    W = modify()
                    # all element of x becomes nan
                except Exception as e:
                    break

            xs.append(x)

        xs = np.array(xs, dtype=bool)

        return (xs[0:self.scout, :], xs[self.scout:, :])

    def evaluation(self, iter):
        'evaluation for employed/scout bees and make onlooker/scout bees'

        bees = np.concatenate((self.EmployedBees, self.ScoutBees), axis=0)
        F = bees.dot(self.profit)
        F = F.astype(np.float128)

        # scout bees
        ansS = []
        if iter != 1:
            tmpbees = np.concatenate((bees, self.OnLookerBees), axis=0)
        else:
            tmpbees = bees
        score = tmpbees.dot(self.profit)
        score = score.astype(np.float32)
        for i in range(self.scout):
            idx = np.nanargmax(score)
            score[idx] = np.nan
            # scout bees are not selected by onlooker bees
            if idx < self.employed + self.scout:
                F[idx] = 0
            ansS.append(tmpbees[idx])
        ansS = np.array(ansS, dtype=bool)

        # onlooker bees
        F_mu = np.power(F, self.mu)

        # Ch = F_mu / np.sum(F_mu)

        bound = [np.sum(F_mu[0:i+1])/ np.sum(F_mu) for i in range(F_mu.size)]
        bound = np.array(bound)
        # print(bound)
        rs = np.random.random(size=self.onlooker)
        idxs = []
        for r in rs:
            # the index that becomes true at first indicates region that contains the value
            idxs.append(np.argmax(r < bound))

        ansO = []
        # onlooker bees index
        for i in idxs:
            ansO.append(bees[i])
        ansO = np.array(ansO, dtype=bool)
        return (ansO, ansS)


    def selection(self):
        'selection by scout bees and recruiting'
        # self.beta * pref / self.H + self.gamma
        arg = np.argsort(self.ScoutBees.dot(self.profit))

        arr = np.zeros(self.EmployedBees.shape, dtype=bool)
        region = np.arange(1, 1+self.scout) * self.beta / self.H + self.gamma
        region = [np.sum(region[0:i]) for i in range(1, self.scout+1)]
        region = np.array(list(map(int, region)))

        for j in range(self.employed):
            idx = np.argmax(j < region)
            arr[j] = self.ScoutBees[arg[idx]]

        return arr

    def search(self, iter):
        'search neighborhood of EmployedBees and OnLookerBees'
        p_change = self.p_min + iter/self.maxIter * (self.p_max - self.p_min)
        tmp = np.concatenate((self.EmployedBees, self.OnLookerBees), axis=0)
        arr = tmp.copy()

        changeFlag = np.random.rand(self.employed+self.onlooker, self.size) < p_change
        ans = np.bitwise_xor(changeFlag, arr)

        arg = np.argsort(self.relation)
        # Some bees exceed C.
        for i in range(self.employed+self.onlooker):
            if ans[i].dot(self.weight) > self.C:
                # remove item which have less quaility.
                idx = 0
                while ans[i].dot(self.weight) > self.C:
                    ans[i][arg[idx]] = False
                    idx += 1

        return (ans[0:self.employed, :], ans[self.employed:, :])

    def main(self):
        'maximize p.dot(x) subject to weight.dot(x) <= C'
        # step1 __init__

        # step2
        self.ScoutBees, self.EmployedBees = self.initAns()

        best_solution = []
        ave = []
        iter = 1
        while True:
            # step3 (OnLookerBees take an action, watching EmployedBees/ScoutBees behavior. Decide ScoutBees here.)
            self.OnLookerBees, self.ScoutBees = self.evaluation(iter)
            # step4 (Register)
            bees = np.concatenate((self.EmployedBees, self.OnLookerBees, self.ScoutBees), axis=0)
            best_solution.append(bees[np.argmax(bees.dot(self.profit))])
            ave.append(np.mean(bees.dot(self.profit)))
            # step5 (assign EmployedBees to ScoutBees)
            self.EmployedBees = self.selection()
            # step6
            if iter == self.maxIter:
                break
            # step7 (EmployedBees/OnLookerBees search around there.)
            self.EmployedBees, self.OnLookerBees = self.search(iter)
            # step8
            iter += 1

        best_solution = np.array(best_solution, dtype=np.int32)
        print(best_solution[-1])
        print(best_solution.dot(self.profit).tolist())
        print(ave)
        ave = np.array(ave)

def makeBin(arr, repeat):
    if repeat == 1:
        return arr
    else:
        s = arr.shape[0]
        z = np.hstack((np.zeros((s, 1), dtype=bool), arr))
        o = np.hstack((np.ones((s, 1), dtype=bool), arr))
        return makeBin(np.vstack((z, o)), repeat-1)

def answer(size, weight, profit, C):
    if size >= 23:
        return
    l = makeBin(np.array([[False], [True]]), size)

    pr = l.dot(profit)
    pr = (l.dot(weight) <= C) * pr
    idx = np.argmax(pr)
    print(l[idx].astype(np.int32))
    print(l[idx].dot(profit).tolist())

if __name__ == '__main__':
    # N <= 22
    # N = 22
    for datanum in range(10):
        f = open("dataset.txt", "r")
        for i in range(8):
            print("problem{}".format(i+1))
            c = int(f.readline())
            w = f.readline()
            w = list(map(int, w[1:-2].split(", ")))
            p = f.readline()
            p = list(map(int, p[1:-2].split(", ")))
            ans = f.readline()
            ans = np.array(list(map(int, ans[2:-2].split("\\n")[:-1])))

            N = len(w)
            abc = ABC(size=N)
            abc.weight = np.array(w)
            abc.profit = np.array(p)
            abc.C = c
            # the number of bees is small
            print("param: maxiter=50, bees=(50, 50, 5), H=5")
            abc.set_params(maxiter=50, bees=(50, 50, 5), p=(4e-2, 12e-2), mu=3, beta=-10, gamma=16, H=5)
            abc.main()
            print("param: maxiter=50, bees=(28, 28, 3), H=3")
            abc.set_params(maxiter=50, bees=(28, 28, 3), p=(4e-2, 12e-2), mu=3, beta=-10, gamma=16, H=3)
            abc.main()
            print("param: maxiter=50, bees=(6, 6, 1), H=1")
            abc.set_params(maxiter=50, bees=(6, 6, 1), p=(4e-2, 12e-2), mu=3, beta=-10, gamma=16, H=1)
            abc.main()
            # print("param: maxiter=50, bees=(3, 3, 1), H=1")
            # abc.set_params(maxiter=50, bees=(3, 3, 1), p=(4e-2, 12e-2), mu=3, beta=-10, gamma=16, H=1)
            # abc.main()

            # employed, onlooker don't work.
            print("param: p=(4e-2, 12e-2)")
            abc.set_params(maxiter=50, bees=(50, 50, 5), p=(4e-2, 12e-2), mu=3, beta=-10, gamma=16, H=5)
            abc.main()
            print("param: p=(1e-2, 3e-2)")
            abc.set_params(maxiter=50, bees=(50, 50, 5), p=(1e-2, 3e-2), mu=3, beta=-10, gamma=16, H=5)
            abc.main()
            print("param: p=(0, 0)")
            abc.set_params(maxiter=50, bees=(50, 50, 5), p=(0, 0), mu=3, beta=-10, gamma=16, H=5)
            abc.main()
            print("answer")
            print(ans)
            print(ans.dot(abc.profit))
            # input('If you want to search answer by checking all patterns, press ENTER.\n')
            # by checking all patterns, get answer for this problem.
            # answer(size=N, weight=abc.weight, profit=abc.profit, C=abc.C)
        else:
            f.close()
