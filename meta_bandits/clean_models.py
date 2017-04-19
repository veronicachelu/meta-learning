import os

models = os.listdir('./models')
for model in models:
    dir = os.path.join('./models', model)
    if len(os.listdir(dir)) == 0:
        os.rmdir(dir)
