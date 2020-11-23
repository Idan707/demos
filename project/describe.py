from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem
import mlrun

import pandas as pd

from dask.distributed import Client
from dask import dataframe as dd

def describe(context: MLClientCtx,
                dataset: DataItem,
                dask_address: str='') -> None:
    """
    Simple describe function with dask 
    
    :param context:                 Function context
    :param dataset:                 Raw data file
    :param dask_key:                Dask client address
    """
    
    context.logger.info("Init Dask")
    client = Client(dask_address)
    
    context.logger.info("Read Data")
    
    df = dataset.as_df(df_module=dd)
    df = df.describe().compute()
    
    context.log_dataset("describe", 
                        df=df,
                        format='csv', index=True, 
                        labels={"data-type": "describe"})