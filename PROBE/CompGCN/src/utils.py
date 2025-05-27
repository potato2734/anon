import os
def read_entities(path):
    e2id = {}
    with open(path) as f:
        for line in f:
            eid, e = line.strip().split('\t')
            e2id[e] = int(eid)
    return e2id

def read_relations(path):
    r2id = {}
    with open(path) as f:
        for line in f:
            rid, r = line.strip().split('\t')
            r2id[r] = int(rid)
    return r2id

def read_triple(file_path, entity2id, relation2id):
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def read_triples(path):
    triples = []
    with open(path) as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            triples.append((h,r,t))
    return triples