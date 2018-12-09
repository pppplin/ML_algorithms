import sys, time

class DecisionTreeNode:
    def __init__(self, attr = None, val = None):
        self.attr = attr
        self.val = val
        self.result = None
        self.next_level = None

class DecisionTree:
    def __init__(self, root):
        self.root = root
        self.label = None
        self.min_attr = None
        self.attr_dict = None
        self.attr_list = None
        self.data = None

    def load_data(self, fn):
        f = open(fn, 'r')
        data = [l.replace('\n', '') for l in f]
        self.data = data

    def process_data(self, data):
        '''
        data: list of data strings
        return type: number of attributes, values in each attribute
        '''
        num_of_attr = len(data[0].split(' ')) - 1
        all_attr = [int(d[0]) for d in data[0].split(' ')[1: ]]
        self.min_attr = min(all_attr)

        self.attr_dict = {}
        for attr in range(self.min_attr, self.min_attr + num_of_attr):
            temp = [int(i.split(' ')[attr - self.min_attr + 1][-1]) for i in data]
            self.attr_dict[attr] = max(temp)
        self.attr_list = list(self.attr_dict.keys())
        self.label = max([int(i[0][0]) for i in data])

    def _gini(self, data, attr):
        '''
        attr: under the given attribute
        return type: gini index
        '''
        class_dict = {}
        inner_dict = {}
        for v in range(1, self.attr_dict[attr] + 1):
            for l in range(1, self.label + 1):
                inner_dict[(v, l)] = 0
            class_dict[v] = 0

        for d in data:
            v = int(d.split(' ')[attr - self.min_attr + 1][-1])
            l = int(d[0])
            class_dict[v] += 1
            inner_dict[(v, l)] += 1

        s = 0
        for v in range(1, self.attr_dict[attr] + 1):
            t = 0
            for l in range(1, self.label + 1):
                if class_dict[v] > 0:
                    inner_dict[(v, l)] /= class_dict[v]
                    t += inner_dict[(v, l)]**2
            s += (1 - t) * (class_dict[v]/len(data))
        return s

    def _split_data(self, data, attr, val):
        result = [d for d in data if int(d.split(' ')[1: ][attr - self.min_attr][-1]) == val]
        return result

    def _majority_vote(self, data):
        d = {}
        for i in range(self.label + 1):
            d[i] = 0
        for l in data:
            d[int(l[0])] += 1

        max_val = d[0]
        max_label = 0
        for i in range(1, self.label + 1):
            if d[i] > max_val:
                max_val = d[i]
                max_label = i
        return max_label

    def _find_next_level(self, data):
        min_gini_attr, min_gini = None, None
        for attr in self.attr_list:
            cur_gini = self._gini(data, attr)
            if not min_gini or cur_gini < min_gini:
                min_gini = cur_gini
                min_gini_attr = attr

        if min_gini == 0:
            result = int(data[0][0])
        else:
            result = [DecisionTreeNode(attr = min_gini_attr, val = v) for v in range(1, self.attr_dict[min_gini_attr] + 1)]
        return result

    def build(self, root, data, level):
        next_level = self._find_next_level(data)
        level += 1
        if type(next_level) is list:
            root.next_level = next_level
            for node in next_level:
                updated_data = self._split_data(data, node.attr, node.val)
                if not updated_data:
                    node.result = self._majority_vote(data)
                elif len(data) == len(updated_data):
                    node.result = self._majority_vote(updated_data)
                elif self._check_labels(updated_data):
                    node.result = int(updated_data[0][0])
                else:
                    node = self.build(node, updated_data, level)
        else:
            root.result = next_level

    def _check_labels(self, data):
        label = int(data[0][0])
        for d in data:
            if int(d[0]) != label:
                return False
        return True

    def test(self, data, root):
        overall = len(data)
        correct = 0
        for d in data:
            if self.predict(d, root):
                correct += 1
        print("correct %d"%correct, 'overall %d'%overall, 'accuracy ', correct/overall)

    def predict(self, data, node):
        while not node.result:
            for n in node.next_level:
                if int(data.split(' ')[n.attr - self.min_attr + 1][-1]) == n.val:
                    node = n
                    break

        #if int(data[0]) != int(node.result):
        #    print('data is ', data[0], ' predict is ', node.result)
        return int(data[0]) == int(node.result)

    def _visualization(self, root, level = 0):
        print('level %d'%level, 'attr %s'%root.attr, 'val %s'%root.val)
        if root.next_level:
            for node in root.next_level:
                new_level = level + 1
                self._visualization(node, new_level)

if __name__ == '__main__':
    t = time.time()
    root = DecisionTreeNode()
    tree = DecisionTree(root)
    tree.load_data(sys.argv[1])
    tree.process_data(tree.data)
    tree.build(tree.root, tree.data, 0)

    tree.load_data(sys.argv[2])
    tree.test(tree.data, tree.root)
    print(time.time() - t)

