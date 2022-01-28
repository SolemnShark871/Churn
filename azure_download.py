from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, generate_account_sas, ResourceTypes, AccountSasPermissions
import os
import errno

sas_token = generate_account_sas(
    account_name="loaddatafunc",
    account_key="PBRnQ0j9XFfH9w4QPAFssLwRzWNioWQjo0tzxhPtCnmchLYOOBwZMytQk3EXN0AOB5MHIfT/E/VMlZ2yuxpbRw==",
    resource_types=ResourceTypes(service=True,container=True,object=True),
    permission=AccountSasPermissions(read=True,list=True,write=True,delete=True,create=True,add=True,process=True,delete_previous_version=True,update=True),
    expiry=datetime.utcnow() + timedelta(hours=1)
)

def download_data(engine: str):
    blob_service_client = BlobServiceClient(account_url="https://loaddatafunc.blob.core.windows.net/churn-files", credential=sas_token)

    container_client = blob_service_client.get_container_client("churn-files")

    generator = container_client.list_blobs(name_starts_with="downloads")

    for blob in generator:
        if not os.path.exists(os.path.dirname(blob.name)):
            try:
                os.makedirs(os.path.dirname(blob.name))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise


        with open(blob.name, "wb+") as dest:
            stream = container_client.download_blob(blob=blob)
            data = stream.readall()
            dest.write(data)

