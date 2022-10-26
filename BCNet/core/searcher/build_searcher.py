from .uniform_searcher import UniformSearcher, UniformMultiPathSearcher
from .sense_searcher import SenseSearcher, SenseMultiPathSearcher


def build_searcher(searcher_type, **kwargs):
    if searcher_type == 'uniform':
        return UniformSearcher()
    elif searcher_type == 'uniform_multi_path':
        return UniformMultiPathSearcher(**kwargs)
    elif searcher_type == 'sense':
        return SenseSearcher(**kwargs)
    elif searcher_type == 'sense_multi_path':
        return SenseMultiPathSearcher(**kwargs)
