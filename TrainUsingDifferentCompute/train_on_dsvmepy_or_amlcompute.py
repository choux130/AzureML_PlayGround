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
""" 02a -- Configuration (Environment)
"""
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

myenv = Environment.from_conda_specification(name = "myenv", file_path="./myenv.yml")
myenv.docker.enabled = True
myenv.python.user_managed_dependencies = False

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
import shutil

script_folder = './aml-run' # this is the folder that we are going to send to the remote vm
os.makedirs(script_folder, exist_ok=True)
shutil.copy('./train.py', os.path.join(script_folder, 'train.py'))
shutil.copy('./myfucs.py', os.path.join(script_folder, 'myfucs.py'))

#%% 
""" 02d-1 -- Configuration (Compute - DSVM)
"""
from azureml.core.compute import RemoteCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException

dsvm_name = "vm-1"
target_compuate = "attach-vm-1"
try:
    target_compuate = RemoteCompute(workspace=ws, name=target_compuate)
    print('found existing:', target_compuate.name)
except ComputeTargetException:
    vm_username = os.environ.get("vm_user_name",  default = 'ting')
    vm_password = os.environ.get("vm_password", default = 'Password123!')
    attach_config = RemoteCompute.attach_configuration(
        resource_id='/subscriptions/' + ws.subscription_id + '/resourceGroups/' + ws.resource_group + '/providers/Microsoft.Compute/virtualMachines/' + dsvm_name,
        ssh_port=22,
        username=vm_username,
        password=vm_password)
    target_compuate = ComputeTarget.attach(workspace=ws,
                                           name=target_compuate,
                                           attach_configuration=attach_config)
    target_compuate.wait_for_completion(show_output=True)

#%% 
""" 02d-2 -- Configuration (Compute - AML)
"""
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

cpu_cluster_name = "cpu-cluster"
# Verify that cluster does not exist already
try:
    target_compuate = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2', max_nodes=4)
    target_compuate = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

target_compuate.wait_for_completion(show_output=True)

#%%
""" 02 -- Configuration (All)
"""
from azureml.core.runconfig import RunConfiguration
from azureml.core import ScriptRunConfig
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

run_config = RunConfiguration()
run_config.target = target_compuate
run_config.environment = myenv
run_config.data_references = {ds.name: dr}

script_arguments = ['--alpha_start', 0, 
                    '--alpha_end', 4, 
                    '--alpha_by', 0.5, 
                    '--data-folder', str(ds.as_download())]
config = ScriptRunConfig(source_directory='./aml-run', 
                         script='train.py',
                         run_config = run_config,
                         arguments = script_arguments)

#%%
""" 03 -- Run the experiment
"""
experiment_name = 'train-on-remote-vm'
exp = Experiment(workspace=ws, name=experiment_name)
run = exp.submit(config, tags = {"tag": "remote_dev"})
run.wait_for_completion(show_output = True)
