import numpy as np

class ABC(object):
    """docstring for ABC
    solving KP by using ABC algorithm
    """
    def __init__(self, size=10):
        'set datasize, data(problem) and set parameters for ABC'
        self.size = size
        self.getdata()
        self.set_params(maxiter=1)

    def getdata(self):
        'set a problem'
        self.profit = np.random.randint(1, 50, self.size)
        self.weight = np.random.randint(1, 30, self.size)
        self.C = np.random.randint(1 * self.size, 20 * self.size)

    def set_params(self, maxiter=50, bees=(50, 50, 5), p=(4e-2, 12e-2), mu=3, beta=-10, gamma=16, H=5):
        'set parameters'
        self.maxIter = 1
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
        relation = self.profit / self.weight
        T = np.sum(self.weight)
        prob = self.C / T * relation / np.mean(relation)

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

        xs = []
        for i in range(self.employed):
            x = np.random.rand(self.size) < relation
            # print(x)
            W = self.weight.dot(x)
            # print(W)
            while self.C < W:
                try:
                    W = modify()
                    # all element of x becomes nan
                except Exception as e:
                    break

            xs.append(x)

        xs = np.array(xs)
        # print(xs.shape)
        return xs

    def evaluation(self):
        'evaluation for employed bees and make onlooker bees'
        F = self.EmployedBees.dot(self.profit)
        F_mu = np.power(F, self.mu)
        Ch = F_mu / np.sum(F_mu)
        # print(Ch)
        bound = [np.sum(F_mu[0:i+i])/ np.sum(F_mu) for i in range(F_mu.size)]
        bound = np.array(bound)
        # print(bound)
        rs = np.random.random(size=F_mu.size)
        idx = []
        for r in rs:
            # the index that becomes true at first indicates region that contains the value
            idx.append(np.argmax(r < bound))

        ans = []
        # onlooker bees index
        for i in idx:
            ans.append(self.EmployedBees[i])
        return np.array(ans)


    def selection(self):
        'selection of scout bees and recruiting'
        self.beta * pref / self.H + self.gamma

    def search(self, iter):
        'search neighborhood'
        p_change = self.p_min + iter/self.maxIter * (self.p_max - self.p_min)

    def main(self):
        'maximize p.dot(x) subject to weight.dot(x) <= C'
        # step1 __init__

        # step2
        self.EmployedBees = self.initAns()
        print(self.EmployedBees.shape)
        # print(self.profit)
        # print(self.weight)
        # print(self.EmployedBees)
        # print(self.C, ">=", self.EmployedBees.dot(self.weight))
        # print(self.EmployedBees.dot(self.profit))

        best_soluation = []
        iter = 1
        while True:
            # step3
            self.OnLookerBees = self.evaluation()
            print(self.OnLookerBees.shape)
            # step4
            best_soluation.append(self.EmployedBees[np.argmax(self.EmployedBees.dot(self.profit))])
            # step5
            self.selection()
            # step6
            if iter == self.maxIter:
                break
            # step7
            self.search(iter)
            # step8
            iter += 1

if __name__ == '__main__':
    abc = ABC()
    abc.main()
