def import_name(module_name: str, name: str):
    """Import a named object from a module in the context of this function."""
    module = __import__(module_name, globals(), locals(), [name])
    return vars(module)[name]
