from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, generate_account_sas, ResourceTypes, AccountSasPermissions
import os
import errno 

local_path = "./downloads"
local_file_name = "prueba.txt"
upload_file_path = os.path.join(local_path, local_file_name)


sas_token = generate_account_sas(
    account_name="loaddatafunc",
    account_key="PBRnQ0j9XFfH9w4QPAFssLwRzWNioWQjo0tzxhPtCnmchLYOOBwZMytQk3EXN0AOB5MHIfT/E/VMlZ2yuxpbRw==",
    resource_types=ResourceTypes(service=True,container=True,object=True),
    permission=AccountSasPermissions(read=True,list=True,write=True,delete=True,create=True,add=True,process=True,delete_previous_version=True,update=True),
    expiry=datetime.utcnow() + timedelta(hours=1) )



blob_service_client = BlobServiceClient(account_url="https://loaddatafunc.blob.core.windows.net", credential=sas_token )
blob_client = blob_service_client.get_blob_client(container="churn-files/downloads", blob=local_file_name)

# Upload the created file
with open(upload_file_path, "rb") as data:
    blob_client.upload_blob(data)