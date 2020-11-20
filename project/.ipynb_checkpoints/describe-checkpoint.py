from mlrun.execution import MLClientCtx
from mlrun.datastore import DataItem

import pandas as pd

from dask.distributed import Client
from dask import dataframe as dd

def describe(context: MLClientCtx,
                dataset: DataItem,
                dask_address: DataItem) -> None:
    """
    Simple describe function with dask 
    
    :param context:                 Function context
    :param dataset:                 Raw data file
    :param dask_key:                Dask client address
    """
    
    context.logger.info("Init Dask")
    
    address = dask_address.as_df(df_module=pd)
    address = address.client[0]
    client = Client(address)
    
    context.logger.info("Read Data")
    
    df = dataset.as_df(df_module=dd)
    df = df.describe().compute()
    
    context.log_dataset("describe", 
                        df=df,
                        format='csv', index=True, 
                        labels={"data-type": "describe"})