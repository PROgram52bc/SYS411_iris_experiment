import pprint
import csv
import random
import itertools

def proportion_join(data, spec):
    """join a list of lists according to the spec

    :data: a list of lists
    :spec: a list of integers, whose sum must equal to the length of data
    :returns: a generator of iterables whose length equals the length of spec

    """
    if sum(spec) != len(data):
        raise Exception("sum of spec must equal to the length of data")
    for n in spec:
        yield list(itertools.chain(*data[:n]))
        data = data[n:]


def fragment(data, spec):
    """randomly split elements in the list according to the specification

    :data: a list of items
    :spec: either a list of positive integers (Not implemented), or a single integer
        if spec is a list of positive integers [N1, N2, ...],
            the result will be a series of lists with sizes proportional to each integer.
        if sepc is a single integer N, the result will be data equally splitted into N subsets.
    :returns: a nested list containing fragmented data

    """
    def _fragment(data, n):
        """fragment function that accepts only an integer

        :data: a list of items
        :n: the number of fragments to split the data into
        :returns: a nested list containing fragmented data

        """
        samples = list(data)
        random.shuffle(samples)
        return [ samples[i::n] for i in range(n) ]

    if isinstance(spec, int):
        return _fragment(data, spec)
    elif isinstance(spec, list):
        f = _fragment(data, sum(spec))
        return list(proportion_join(f, spec))
        # raise NotImplementedError("current version does not support using list as spec")
    else:
        raise Exception("spec must be either an integer or a list!")


def stratify(data, key):
    """stratify the data based on values retrieved from the key function

    :data: an array
    :key: a function that turns the array element into a hashable object
    :returns: a dictionary with stratified data

    """
    dic = {}
    for v in data:
        k = key(v)
        if k not in dic:
            dic[k] = []
        dic[k].append(v)
    return dic


def main():
    pp = pprint.PrettyPrinter(indent=2)
    data = []
    with open("../data/Iris.csv") as f:
        reader = csv.reader(f)
        next(reader) # skip first line
        for row in reader:
            data.append(row)

    data = stratify(data, lambda c:c[5]).values() # stratify according to the category name (e.g. Iris-setosa) at index 5

    data = [ fragment(category, 5) for category in data ] # randomly fragment each category into 5 sections

    data = [ list(itertools.chain(*sections)) for sections in zip(*data) ] # join each section across the categories

    for item in data:
        random.shuffle(item) # shuffle items

    for i, section in enumerate(data):
        print("Section {}:".format(i))
        pp.pprint(section)
        stratified = stratify(section, lambda c:c[5]) # stratify according to the category name (e.g. Iris-setosa) at index 5
        for key in stratified:
            count = len(stratified[key])
            print("Count of {}: {}".format(key, count))
        print()

if __name__ == "__main__":
    main()
