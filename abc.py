import numpy as np

class ABC(object):
    """docstring for ABC"""
    def __init__(self, size=10):
        self.size = size
        self.getdata()
        self.set_params(maxiter=1)

    def getdata(self):
        self.profit = np.random.randint(1, 50, self.size)
        self.weight = np.random.randint(1, 30, self.size)
        self.C = np.random.randint(1 * self.size, 20 * self.size)

    def set_params(self, maxiter=50, bees=(50, 50, 5), p=(4e-2, 12e-2), mu=3, beta=-10, gamma=16, H=5):
        self.maxIter = 1
        self.employ = bees[0]
        self.onlooker = bees[1]
        self.scout = bees[2]
        self.p_min = p[0]
        self.p_max = p[1]
        self.mu = mu
        self.beta = beta
        self.gamma = gamma
        self.H = H

    def initAns(self):
        relation = self.profit / self.weight
        T = np.sum(self.weight)
        prob = self.C / T * relation / np.mean(relation)
        x = np.random.rand(self.size) < relation
        W = self.weight.dot(x)

        def modify():
            'in x, discard the worse items'
            tmp = relation * x
            np.place(tmp, tmp==0.0, np.nan)
            try:
                idx = np.nanargmin(tmp)
            except Exception as e:
                raise e
            x[idx] = False
            return self.weight.dot(x)

        while self.C < W:
            try:
                W = modify()
            # all element of x becomes nan
            except Exception as e:
                break

        return x

    def evaluation(self):
        pass
    def selection(self):
        pass
    def get_ans(self):
        pass
    def selection(self):
        pass
    def search(self):
        pass

    def main(self):
        'maximize p.dot(x) subject to weight.dot(x) <= C'
        ans = self.initAns()
        print(self.profit)
        print(self.weight)
        print(self.C)
        print(ans)
        best_soluation = []
        for iter in range(self.maxIter):
            self.evaluation()
            best_soluation.append(self.get_ans())
            self.selection()
            self.search()

if __name__ == '__main__':
    abc = ABC()
    abc.main()
