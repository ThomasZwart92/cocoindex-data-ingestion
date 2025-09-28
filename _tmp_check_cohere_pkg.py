import importlib.util
spec = importlib.util.find_spec('cohere')
print('cohere installed:', bool(spec))
