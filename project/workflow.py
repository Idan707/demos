from kfp import dsl
from mlrun import mount_v3io

funcs    = {}
LABELS   = "label"
DATA_URL = "/User/iris.csv"
#DATA_URL = "/User/yellow_tripdata_2019-01_subset.csv"


# init functions is used to configure function resources and local settings
def init_functions(functions: dict, project=None, secrets=None):
    for f in functions.values():
        f.apply(mount_v3io())
        pass
     
    
@dsl.pipeline(
    name="Demo training pipeline",
    description="Shows how to use mlrun."
)
def kfpipeline():
    
    # init_dask
    dask_init = funcs['dsjob'].as_step(
        handler="hndlr",
        params={"client_url" : "db://default/mydask"},
        outputs=['client'])
    
    # describe data
    describe = funcs['describe'].as_step(
        handler="describe",
        inputs={"dataset"   : DATA_URL,
                "dask_address"  : dask_init.outputs['client']})
    
    # get data, train, test and evaluate 
    train = funcs['dask_classifier'].as_step(
        name="train-skrf",
        handler="train_model",
        params={"label_column"    : LABELS,
                "test_size"       : 0.10,
                "model_pkg_class" : "sklearn.ensemble.RandomForestClassifier"},
        inputs={"dataset"   : DATA_URL,
                "dask_address"  : dask_init.outputs['client']},
        outputs=['model', 'test_set'])
