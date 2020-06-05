#%%
""" 00 -- Confirm the version of the package azureml.core 
"""
import azureml.core
print("SDK version:", azureml.core.VERSION)

#%%
""" 01a -- Connect to the Workspace
"""
import os
from azureml.core import Workspace, Experiment, Datastore, Dataset
from azureml.core.authentication import ServicePrincipalAuthentication

# this is the most important credentials, should be saved as an environment variable. 
# https://github.com/Azure/MachineLearningNotebooks/blob/1f05157d24c8bd9866121b588e75dc95764ae898/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb
svc_pr_password = os.environ.get("AZURE_CLIENT_SECRET", default = 'xxx')
svc_pr = ServicePrincipalAuthentication(
    tenant_id="xxx",
    service_principal_id="xxx",
    service_principal_password=svc_pr_password)
ws = Workspace(
    subscription_id = 'xxx', 
    resource_group = 'xxx', 
    workspace_name = 'xxx',
    auth=svc_pr)  

#%%
""" 01b -- Upload data files into default storage (only need to do once)
"""
# ds = ws.get_default_datastore()
# print(ds.name, ds.datastore_type, ds.account_name, ds.container_name)

# from sklearn.datasets import load_diabetes
# import numpy as np

# training_data = load_diabetes()
# np.save(file='./features.npy', arr=training_data['data'])
# np.save(file='./labels.npy', arr=training_data['target'])
# ds.upload_files(['./features.npy', './labels.npy'], target_path='diabetes', overwrite=True)

#%%
""" 02a -- Configuration (Environment)
"""
# https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.environment(class)?view=azure-ml-py
# skip, if using current environment
# later, make sure you set run_config.environment.python.user_managed_dependencies = True 

#%%
""" 02b -- Configuration (Data Reference)
"""
ds = ws.get_default_datastore()
print(ds.name, ds.datastore_type, ds.account_name, ds.container_name)
dataset = Dataset.File.from_files(path = [(ds, 'diabetes/')])
print(dataset.to_path())

from azureml.core.runconfig import DataReferenceConfiguration
dr = DataReferenceConfiguration(datastore_name=ds.name, 
                   path_on_datastore='diabetes', 
                   path_on_compute='/tmp/azureml_runs',
                   mode='download', 
                   overwrite=True)

#%%
""" 02c -- Configuration (Directory)
"""
# skip, if using local compute

#%% 
""" 02d -- Configuration (Compute)
"""
# skip, if using local compute

#%%
""" 02 -- Configuration (All)
"""
from azureml.core.runconfig import RunConfiguration
from azureml.core import ScriptRunConfig
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

run_config = RunConfiguration()
run_config.environment.python.user_managed_dependencies = True
run_config.data_references = {ds.name: dr}

script_arguments = ['--alpha_start', 0, 
    '--alpha_end', 4, 
    '--alpha_by', 0.5,
    '--data-folder', str(ds.as_download())]
config = ScriptRunConfig(source_directory='./', 
                         script='train.py',
                         run_config = run_config,
                         arguments = script_arguments)

#%%
""" 03 -- Run the experiment
"""
experiment_name = 'train-on-remote-vm'
exp = Experiment(workspace=ws, name=experiment_name)
run = exp.submit(config, tags = {"tag": "local_dev"})
run.wait_for_completion(show_output = True)


# %%
