import pickle


class Candidate(list):

    def __init__(self, n, log_path=None, rank=-1):
        super().__init__()
        self.n = n
        self.total = 0
        self._idx = []
        self.sample_from_net = 0
        self.sample_from_cand = 0
        self.FLAG = False
        self.in_out = 0
        self.logger = None
        self.rank = rank
        if log_path is not None and self.rank == 0:
            self.logger = open(log_path, 'a')
            self.logger.write(f'iter,score,ori_idx,new_idx,total,subnet,flops\n')

    def _add(self, c, flops=0, iter_num=0):
        assert isinstance(c, tuple)
        inserted = False

        # smooth score
        ori_idx = -1
        new_idx = -1
        if c[0] in self._idx:
            t_idx = self._idx.index(c[0])
            ori_idx = t_idx
            c = (c[0], (self[t_idx][1] + c[1]) / 2.)
            del self[t_idx]
            del self._idx[t_idx]
            self.total -= 1

        # insert c
        for i in range(self.total):
            _c_score = self[i][1]
            if c[1] > _c_score:
                self.insert(i, c)
                self._idx.insert(i, c[0])
                self.total += 1
                inserted = True
                new_idx = i
                break

        if not inserted and self.total < self.n:
            # insert to tail
            self.append(c)
            self._idx.append(c[0])
            self.total += 1
            new_idx = self.total - 1

        if inserted and self.FLAG and self.total >= self.n:
            self.in_out += 1

        if self.logger is not None:
            self.logger.write(f'{iter_num},{c[1]},{ori_idx},{new_idx},{self.total},{c[0]},{flops}\n')

        if self.total > self.n:
            self.total -= 1
            if self.logger is not None:
                self.logger.write(f'{iter_num},{self[-1][1]},{self.total - 1},{-1},{self.total},{self._idx[-1]},{0}\n')
            del self[-1]
            del self._idx[-1]
  
        if self.logger is not None:
            self.logger.flush()

       
    
    def save(self, save_path):
        logger = self.logger
        self.logger = None
        f = open(save_path, 'wb')
        pickle.dump(self, f)
        f.close()
        self.logger = logger

    @staticmethod
    def load(load_path):
        f = open(load_path, 'rb')
        cand = pickle.load(f)
        f.close()
        return cand
