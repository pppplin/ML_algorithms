import sys, time
from DecisionTree import DecisionTreeNode, DecisionTree
from random import seed, choice
from collections import defaultdict

class RandomForest():
    def __init__(self, N, p):
        '''
        N: number of trees
        p: range from [0, 1], portion of data to choose for each tree
        '''
        self.N = N
        self.p = p
        self.data = None
        self.forest = []

    def load_data(self, fn):
        f = open(fn, 'r')
        data = [l.replace('\n', '') for l in f]
        self.data = data

    def _random_data_blocks(self):
        seed(2)
        #random.randint(101)
        pn = int(self.p * len(self.data))
        L = len(self.data)
        selected_data = []
        selected_num = []
        for j in range(self.N):
            selected_block = []
            for i in range(pn):
                selected = choice(range(L))
                selected_num.append(selected)
                selected_block.append(self.data[selected])
            selected_data.append(selected_block)
        return selected_data

    def build(self):
        for i in range(self.N):
            root = DecisionTreeNode()
            self.forest.append(DecisionTree(root))

        data_blocks = self._random_data_blocks()
        for i in range(len(self.forest)):
            tree = self.forest[i]
            tree.data = data_blocks[i]
            tree.process_data(self.data)
            tree.build(tree.root, tree.data, 0)

    def majority_vote(self, result):
        max_num = 0
        max_label = None
        d = defaultdict(int)
        for i in result:
            d[i] += 1
            if d[i] > max_num:
                max_num = d[i]
                max_label = i
        return max_label

    def test(self):
        overall = len(self.data)
        correct = 0
        for d in self.data:
            result = self.predict(d)
            voted_result = self.majority_vote(result)
            if int(voted_result) == int(d[0]):
                correct += 1
        print("correct %d"%correct, 'overall %d'%overall, 'accuracy ', correct/overall)

    def predict(self, data):
        result = []
        for tree in self.forest:
            node = tree.root
            while not node.result:
                for n in node.next_level:
                    if int(data.split(' ')[n.attr - tree.min_attr + 1][-1]) == n.val:
                        node = n
                        break
            result.append(node.result)
        return result

if __name__ == '__main__':
    t = time.time()
    N = 30
    p = 0.8
    forest = RandomForest(N, p)
    forest.load_data(sys.argv[1])
    forest.build()

    forest.load_data(sys.argv[2])
    forest.test()
    print(time.time() - t)
