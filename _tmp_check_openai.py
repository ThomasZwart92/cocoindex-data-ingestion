import openai
print('openai version =', getattr(openai,'__version__','?'))
print('has OpenAI class =', hasattr(openai,'OpenAI'))
print('has chat attr =', hasattr(openai,'chat'))
print('has ChatCompletion =', hasattr(openai,'ChatCompletion'))
