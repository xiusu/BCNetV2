import numpy as np
from core.sampler.evolution import nsganet as engine
from core.search_space.ops import channel_mults

from pymop.problem import Problem
from pymoo.optimize import minimize
from core.sampler.base_sampler import BaseSampler
from core.sampler.sampler_model import Sampler_extension, Sampler_1Dto2D
from core.utils.arch_util import _decode_arch
import cellular.pape.distributed as dist
import torch
import copy


class EvolutionSampler(BaseSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate_subnet(self):
        return None
        return _decode_arch(self.subnet_topk)  # TODO: fix bug

    def eval_subnet_host(self, subnet):
        finished = torch.Tensor([0])
        dist.broadcast(finished, 0)  # not finished

        # broadcast subnet, then run together
        dist.broadcast(torch.Tensor(subnet), 0)
        score = self.eval_subnet(subnet)
        return score

    def sample(self, sampling=None):
        '''
        @sampling: initial population: 2d numpy array, dtype:np.int
        '''
        subnet_eval_dict = {}
        if self.rank == 0:
            if getattr(self,'Var_len'):
                #n_var = sum(self.bin_number_list)
                n_var = self.Var_len
            else:
                raise RuntimeError('n_var did not defined in evolution_sampler.py')
            n_offspring = None #40

            # setup NAS search problem
            #n_var = len(self.model.net) * 2  # contains channel search idxes
            lb = np.zeros(n_var)  # left index of each block
            ub = np.zeros(n_var) + len(self.P_train) - 1  # right index of each block
            #for idx, block in enumerate(self.model.net):
            #    ub[idx] = len(block) - 1
            #    if getattr(block[0], 'channel_search', False):
            #        ub[idx+len(self.model.net)] = len(channel_mults) - 1

            # this place need to be changed(n_var variable, n_obj objection number, n_constr constraint number, lb low bound, ub up bound, self.eval_subnet_host obj function)

            nas_problem = NAS(n_var=n_var, n_obj=1, n_constr=0, lb=lb, ub=ub,
                              eval_func=lambda subnet: self.eval_subnet_host(subnet),
                              result_dict=subnet_eval_dict)

            # pop_size(how many to be considered together) and n_gens(how many generations to stop) need to be changed
            # configure the nsga-net method
            if sampling is not None:
                method = engine.nsganet(pop_size=self.pop_size,
                                        n_offsprings=n_offspring,
                                        eliminate_duplicates=True,
                                        sampling=sampling)
            else:
                method = engine.nsganet(pop_size=self.pop_size,
                                        n_offsprings=n_offspring,
                                        eliminate_duplicates=True)

            res = minimize(nas_problem,
                           method,
                           callback=lambda algorithm: self.generation_callback(algorithm),
                           termination=('n_gen', self.n_gens))
        else:
            # slaver: wait for signal
            while True:
                finished = torch.Tensor([0])
                dist.broadcast(finished, 0)
                if finished[0] == 1:
                    break

                # get subnet
                subnet = torch.zeros([self.Var_len])
                dist.broadcast(subnet, 0)

                subnet = [int(x) for x in np.array(subnet)]
                self.eval_subnet(subnet)

        # finished
        if self.rank == 0:
            finished = torch.Tensor([1])
            dist.broadcast(finished, 0)

        if self.rank == 0:
            sorted_subnet = sorted(subnet_eval_dict.items(), key=lambda i: i[1], reverse=True)
            sorted_subnet_key = [x[0] for x in sorted_subnet]
            subnet_topk = sorted_subnet_key[:self.sample_num]
            if self.rank == 0:
                print('== search result ==')
                print(sorted_subnet)
                print('== best subnet ==')
                print(subnet_topk)
            self.subnet_topk = subnet_topk

            #str to list
            sample_flist = self.Str_to_list(sorted_subnet_key[:1][0])
            '''
            sample_flist = []
            #print("sorted_subnet_key is {}, length is {}".format(sorted_subnet_key[:1], len(sorted_subnet_key[:1][0])))

            for i in range(len(sorted_subnet_key[:1][0])):
                #print("each pixel is {}".format(sorted_subnet_key[:1][0][i]))

                if sorted_subnet_key[:1][0][i] == '0':
                    sample_flist.append(0)
                elif sorted_subnet_key[:1][0][i] == '1':
                    sample_flist.append(1)
                elif sorted_subnet_key[:1][0][i] == '2':
                    sample_flist.append(2)
                elif sorted_subnet_key[:1][0][i] == '3':
                    sample_flist.append(3)
                elif sorted_subnet_key[:1][0][i] == '4':
                    sample_flist.append(4)
                elif sorted_subnet_key[:1][0][i] == '5':
                    sample_flist.append(5)
                elif sorted_subnet_key[:1][0][i] == '6':
                    sample_flist.append(6)
                elif sorted_subnet_key[:1][0][i] == '7':
                    sample_flist.append(7)
                elif sorted_subnet_key[:1][0][i] == '8':
                    sample_flist.append(8)
                elif sorted_subnet_key[:1][0][i] == '9':
                    sample_flist.append(9)
            #print("sorted_subnet_key is {}".format(sample_flist))
            #assert 1 == 2
            #sample_flist = Sampler_extension(self.skip_list, sorted_subnet_key[:1])
            '''
            sample_flist = torch.IntTensor(sample_flist)

        else:
            sample_flist = torch.zeros([self.Var_len], dtype=torch.int32)

        dist.broadcast(sample_flist, 0)
        sample_flist = sample_flist.tolist()

        Prob_list = []
        for i in range(len(sample_flist)):
            Prob_list.append(self.P_train[sample_flist[i]])

        #sample_flist, _ = self.Channel_list(self.skip_list, sample_flist) #self.bin_list(sample_flist, self.Var_len, self.bin_size_list)
        self.subnet_top1 = Prob_list

        self.subnet_channels_top1 = [int(i * j) for i, j in zip(Prob_list, self.Model_Channels)]
        self.subnet_channels_top1 = Sampler_extension(self.skip_list, self.subnet_channels_top1)

        if self.rank == 0:
            print("Best subnet is {}".format(self.subnet_channels_top1))

        #assert len(sample_flist) == len(self.Model_Channels), "subnet_channels len is {}, num is {}, Model_Channels len is {}, num is {}".format(len(sample_flist), sample_flist, len(self.Model_Channels), self.Model_Channels)

    def Str_to_list(self, c):
        sample_flist = []
        Flag = False
        for i in range(len(c)):
            # print("each pixel is {}".format(sorted_subnet_key[:1][0][i]))
            if c[i] == '0':
                if Flag == True:
                    sample_flist[-1] = sample_flist[-1] * 10 + 0
                else:
                    sample_flist.append(0)
                Flag = True
            elif c[i] == '1':
                if Flag == True:
                    sample_flist[-1] = sample_flist[-1] * 10 + 1
                else:
                    sample_flist.append(1)
                Flag = True
            elif c[i] == '2':
                if Flag == True:
                    sample_flist[-1] = sample_flist[-1] * 10 + 2
                else:
                    sample_flist.append(2)
                Flag = True
            elif c[i] == '3':
                if Flag == True:
                    sample_flist[-1] = sample_flist[-1] * 10 + 3
                else:
                    sample_flist.append(3)
                Flag = True
            elif c[i] == '4':
                if Flag == True:
                    sample_flist[-1] = sample_flist[-1] * 10 + 4
                else:
                    sample_flist.append(4)
                Flag = True
            elif c[i] == '5':
                if Flag == True:
                    sample_flist[-1] = sample_flist[-1] * 10 + 5
                else:
                    sample_flist.append(5)
                Flag = True
            elif c[i] == '6':
                if Flag == True:
                    sample_flist[-1] = sample_flist[-1] * 10 + 6
                else:
                    sample_flist.append(6)
                Flag = True
            elif c[i] == '7':
                if Flag == True:
                    sample_flist[-1] = sample_flist[-1] * 10 + 7
                else:
                    sample_flist.append(7)
                Flag = True
            elif c[i] == '8':
                if Flag == True:
                    sample_flist[-1] = sample_flist[-1] * 10 + 8
                else:
                    sample_flist.append(8)
                Flag = True
            elif c[i] == '9':
                if Flag == True:
                    sample_flist[-1] = sample_flist[-1] * 10 + 9
                else:
                    sample_flist.append(9)
                Flag = True
            else:
                Flag = False
        return sample_flist

    def generation_callback(self, algorithm):
        gen = algorithm.n_gen
        pop_var = algorithm.pop.get("X")
        pop_obj = algorithm.pop.get("F")
        print(f'==Finished generation: {gen}')


# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, n_var=20, n_obj=1, n_constr=0, lb=None, ub=None, eval_func=None, result_dict=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int, )
        self.xl = lb
        self.xu = ub
        self._n_evaluated = 0  # keep track of how many architectures are sampled
        self.eval_func = eval_func
        self.result_dict = result_dict

    def _evaluate(self, x, out, *args, **kwargs):

        objs = np.full((x.shape[0], self.n_obj), np.nan)

        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1

            # all objectives assume to be MINIMIZED !!!!!
            temp = list(x[i])
            if self.result_dict.get(str(temp)) is not None:
                acc = self.result_dict[str(temp)]
            else:
                acc = self.eval_func(x[i])
                self.result_dict[str(temp)] = acc

            print('==evaluation subnet:{} prec@1:{}'.format(str(x[i]), acc))

            objs[i, 0] = 100 - acc  # performance['valid_acc']
            # objs[i, 1] = 10  # performance['flops']

            self._n_evaluated += 1
        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")
    print(gen)

    # print(gen, pop_var, pop_obj)

    # report generation info to files

def main():
    # hyper parameters
    pop_size = 50
    n_gens = 20
    n_offspring = 40

    # setup NAS search problem
    n_var = 20
    lb = np.zeros(n_var)  # left index of each block
    ub = np.zeros(n_var) + 4  # right index of each block

    nas_problem = NAS(n_var=n_var, n_obj=1, n_constr=0, lb=lb, ub=ub)

    # configure the nsga-net method
    method = engine.nsganet(pop_size=pop_size,
                            n_offsprings=n_offspring,
                            eliminate_duplicates=True)

    res = minimize(nas_problem,
                   method,
                   callback=do_every_generations,
                   termination=('n_gen', n_gens))
    print(dir(res))
    print(len(res.pop))
    for pop in res.pop[:10]:
        print(pop.F, pop.X)
    return res


if __name__ == "__main__":
    main()
