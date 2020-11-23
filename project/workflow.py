from kfp import dsl
from mlrun import mount_v3io

# params
funcs       = {}
LABELS      = "VendorID"
DROP        = 'congestion_surcharge'
#DATA_URL    = "/User/iris.csv"
DATA_URL    = "/User/yellow_tripdata.csv"
DASK_CLIENT = "tcp://mlrun-mydask-ce9d12ee-0.default-tenant:8786"

# init functions is used to configure function resources and local settings
def init_functions(functions: dict, project=None, secrets=None):
    for f in functions.values():
        f.apply(mount_v3io())
        pass
     
@dsl.pipeline(
    name="Demo training pipeline",
    description="Shows how to use mlrun"
)
def kfpipeline():
    
    # describe data
    describe = funcs['describe'].as_step(
        handler="describe",
        params={"dask_address"  : DASK_CLIENT},
        inputs={"dataset"       : DATA_URL}
    )
    
    # get data, train, test and evaluate 
    train = funcs['dask_classifier'].as_step(
        name="train-skrf",
        handler="train_model",
        params={"label_column"    : LABELS,
                "dask_address"    : DASK_CLIENT,
                "test_size"       : 0.10,
                "model_pkg_class" : "sklearn.ensemble.RandomForestClassifier",
                "drop_cols"       : DROP},
        inputs={"dataset"         : DATA_URL},
        outputs=['model', 'test_set']
    )
    
    train.after(describe)
