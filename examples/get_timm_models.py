import timm
from tqdm import tqdm
from timm.models.registry import is_model, is_model_in_modules, model_entrypoint


features_models = []
models = timm.list_models('*effi*')
pbar = tqdm(models)
for model_name in pbar:
    #pbar.set_postfix({'model_name': model_name})
    fn = model_entrypoint(model_name=model_name)
    print(fn.__code__.co_varnames)
    print(fn.__defaults__)
    if 'features_only' in fn.__code__.co_varnames:
        features_models.append(model_name)
print(features_models)