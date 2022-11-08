from bidict import bidict


def get_bidict_from_list(labels: list) -> bidict:
    dic = {}
    for i in range(len(labels)):
        dic[i] = labels[i]
    return bidict(dic)
