from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import logging
import os
from datetime import datetime
import json
from probe import PROBE

def set_logger(args):
    log_file = os.path.join(f'./{args.save}/', f'TuckEr_{args.dataset}_{args.seed}.log')
    if not os.path.exists(f'./{args.save}/'):
        os.makedirs(f'./{args.save}/', exist_ok=True)
 
    # Create a logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)

    # Remove any existing handlers (if rerunning the script in the same session)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add a FileHandler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return log_file  # Return the log file path for reference

class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200, 
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False, 
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0.):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}
        self.best_mrr = 0.0
        self.best_it = 0.0
        self.id2e = None
        self.id2r = None
        
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    
    def evaluate(self, model, data, count_info_dict_trn, eid2count):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))
        rank_per_class = defaultdict(list)
        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])

            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            predictions = model.forward(e1_idx, r_idx)
            
            for j in range(data_batch.shape[0]): # for batch_size(128)
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)
            
            sort_idxs = sort_idxs.cpu().numpy()
            sort_values = sort_values.cpu().numpy()
            

            for j in range(data_batch.shape[0]):
                target_score = predictions[j, e2_idx[j]].item()
                target = e2_idx[j].item()
                ties = np.where(np.isclose(sort_values[j], target_score))[0]

                ranking = float(ties.mean() + 1)            
                rank_per_class[target].append(ranking)
                ranks.append(ranking)

                for hits_level in range(10):
                    if ranking - 1 <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        count_info_dict_tst = {k: len(v) for k, v in rank_per_class.items()}
        met = PROBE(rank_per_class, count_info_dict_trn, count_info_dict_tst, len(eid2count))
        met.set_raw_ranks(rank_per_class)
        alphas, betas = [0.25, 0.5, 1.0, 2.0], [0.8, 0.4, 0.2, 0.0]
        for a in alphas:
            for b in betas:
                logging.info(f'PROBE(alpha={a},beta={b}) : {met.calculate_final_metric(a, b)}')

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))

        total_metric = {}
        total_metric['MR'] = float(np.mean(ranks))
        total_metric['MRR'] = float(np.mean(1./np.array(ranks)))
        total_metric['HITS@1'] = float(np.mean(hits[0]))
        total_metric['HITS@3'] = float(np.mean(hits[2]))
        total_metric['HITS@10'] = float(np.mean(hits[9]))

        logging.info(f'total metric : {total_metric}')
        
                
        return total_metric['MRR']




    def train_and_eval(self):
        print("Training the TuckER model...")
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
        self.id2e = {i:d.entities[i] for i in range(len(d.entities))}
        self.id2r = {i:d.relations[i] for i in range(len(d.relations))}
        
        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))
        
        eid2count = {}
        for i in range(len(self.entity_idxs)):
            eid2count[i] = 0

        for _triple in train_data_idxs:
            h, r, t = _triple
            eid2count[h] += 1
            eid2count[t] += 1
        count_info_dict_trn = dict(sorted(eid2count.items(), key=lambda item : -item[1]))
        
        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())
        
        print("Starting training...")
        for it in range(self.num_iterations):

            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1])  
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))           
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            print(time.time()-start_train)    
            print(np.mean(losses))
            model.eval()
            if not it%50:
                with torch.no_grad():
                    logging.info("<<Validation>>")
                    current_mrr = self.evaluate(model, d.valid_data, count_info_dict_trn, eid2count)
                    if current_mrr > self.best_mrr:
                        self.best_mrr = current_mrr
                        logging.info("New best MRR!!...<<Test>>")
                        self.evaluate(model, d.test_data, count_info_dict_trn, eid2count)
        logging.info('Best validation MRR at step %d : %f', self.best_it, self.best_mrr)

           

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k237", nargs="?",
                    help="Which dataset to use: FB15k, FB15k237, WN18 or WN18RR.")
    parser.add_argument('--save', type=str)
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                    help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    help="Amount of label smoothing.")
    
    parser.add_argument('-seed', default=0, type=int)

    args = parser.parse_args()

    
    set_logger(args)
    
    logging.info('STARTING POINT')
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True 
    seed = args.seed
    logging.info(f'seed : {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 
    d = Data(data_dir=data_dir, reverse=True)
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr, 
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing)
    experiment.train_and_eval()
                

