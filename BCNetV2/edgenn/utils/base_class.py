from collections import OrderedDict
import inspect


class BaseClass:
    def __init__(self):
        self._ignored_keys = []

        # ignore the hyper-parameters in __init__
        self._ignored_params = inspect.getfullargspec(self.__init__).args
        for super_class in self.__class__.__mro__:
            self._ignored_params += inspect.getfullargspec(super_class.__init__).args
        self._ignored_params = set(self._ignored_params)

    def __setattr__(self, key, value):
        if hasattr(self, '_ignored_params'):
            if key in self._ignored_params:
                if not isinstance(value, (int, float, str, bool)):
                    # only ignore params with basic data types
                    self._ignored_params.remove(key)
        object.__setattr__(self, key, value)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        ignored_keys = self._ignored_keys + list(self._ignored_params)
        state_dict = OrderedDict()
        for key, value in self.__dict__.items():
            if key in ignored_keys:
                continue
            if hasattr(value, 'state_dict'):
                sub_state_dict = value.state_dict()
                for k, v in sub_state_dict.items():
                    state_dict[key + '.' + k] = v
            else:
                state_dict[key] = value
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        ignored_keys = self._ignored_keys + list(self._ignored_params)
        cur_key = None
        sub_state_dict = OrderedDict()
        for k, v in state_dict.items():
            key = k.split('.')[0]
            if key != cur_key:
                if len(sub_state_dict) != 0:
                    self.__dict__[cur_key].load_state_dict(sub_state_dict)
                    sub_state_dict = OrderedDict()
                    
                cur_key = key

            if key in self.__dict__:
                if key in ignored_keys:
                    continue
                if hasattr(self.__dict__[key], 'load_state_dict'):
                    sub_state_dict[k[len(key)+1:]] = v 
                else:
                    self.__dict__[k] = v  

        if len(sub_state_dict) != 0:
            self.__dict__[cur_key].load_state_dict(sub_state_dict)
