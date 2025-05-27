import numpy as np
import math


class PROBE():
    def __init__(self, raw_ranks, count_info_dict_trn, count_info_dict_tst, nentity):
        self.raw_ranks = raw_ranks
        self.count_info_dict_trn = count_info_dict_trn
        self.count_info_dict_tst = count_info_dict_tst
        self.inspect_entity_cnt_list = list(self.count_info_dict_trn.values())
        self.nentity = nentity
        self.alpha = None
        self.W = None
        self.beta = None

    def set_raw_ranks(self, _raw_rank_dict):
        self.raw_ranks = _raw_rank_dict

    def set_transform_function(self, alpha, mode='affine'):
        '''
        pre-requisite : self.raw_ranks, self.alpha

        calculation result : self.transformed_ranks
        '''
        self.alpha = alpha
        if mode == 'vanila':
            self.transformed_ranks = [[(1 / x) ** self.alpha for x in v] for k, v in self.raw_ranks.items()]
            return
        elif mode == 'affine':
            A = 1 / (1 - (1 / self.nentity) ** self.alpha)
            self.transformed_ranks = [[A * ((1 / x) ** self.alpha - 1) + 1 for x in v] for k, v in
                                      self.raw_ranks.items()]
            # just change it to numpy(without hindering position informations)?
        else:
            raise Exception(f'No mode supports \'{mode}\'')

    @staticmethod
    def normalize_array(arr):
        total = np.sum(arr)
        assert total > 0.0
        return arr / total

    def set_class_wise_weighting_function(self, beta):
        '''
        pre-requisite : self.count_info_dict_trn

        calculation result : self.W
        '''
        k = 1
        self.beta = beta
        unit_weights, num_problems = [], []
        
        for key_tst in self.count_info_dict_tst:
            unit_weights.append((1 / (self.count_info_dict_trn[key_tst] + k)) ** self.beta)
            num_problems.append(self.count_info_dict_tst[key_tst])
        unit_weights_np = np.array(unit_weights)
        num_problems_np = np.array(num_problems)
  
        assert len(unit_weights_np) == len(num_problems_np)
        self.W = unit_weights_np * num_problems_np
        self.W = self.normalize_array(self.W)
        assert math.isclose(sum(self.W), 1.0, rel_tol=1e-10)

    def calculate_final_metric(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.set_transform_function(alpha)
        self.set_class_wise_weighting_function(beta)

        def calculate_class_mean(transformed_class_ranks):
            return sum(f_r for f_r in transformed_class_ranks) / len(transformed_class_ranks)

        final_metric_value = 0.0
        self.mu_classes = np.array(
            [calculate_class_mean(transformed_ranks) for transformed_ranks in self.transformed_ranks])
        assert len(self.W) == len(self.mu_classes)
        final_metric_value = sum(self.W * self.mu_classes)
        self.final_metric_value = final_metric_value 
        return self.final_metric_value