# Generated by nuclio.export.NuclioExporter

import mlrun
import pandas as pd
from json import dumps
import os

def hndlr(context,
          client_url=''):
    """
    init dask client 
    
    :param context:     Function context.
    :param client_url:  MLrun DB dask client address 
    """
    
    if client_url:
        client = mlrun.import_function(client_url).client
    elif hasattr(context, 'dask_client'):
        client = context.dask_client
    else:
        raise ValueError('no client')
    
    client_save = client.scheduler_info()['address']
    df = pd.DataFrame([client_save], columns=["client"])
    
    context.log_dataset("client", df = df)