def get_recursive_module(root, name):
    now = root
    for mod in name.split("."):
        now = getattr(now, mod)
    return now

def set_recursive_module(root, name, module):
    now = root
    for mod in name.split(".")[:-1]:
        now = getattr(now, mod)
    now.add_module(name.split(".")[-1], module)
    return now
