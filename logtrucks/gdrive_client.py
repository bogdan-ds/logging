import io
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from tqdm.auto import tqdm


class GDriveClient:

    def __init__(self, settings):
        self.settings = settings
        self.scopes = ["https://www.googleapis.com/auth/drive"]
        self.creds = self.authenticate()

    def authenticate(self):
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file(
                'token.json', self.scopes)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', self.scopes)
                creds = flow.run_local_server(port=0)
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        return creds

    def list(self, query=None, page_size=100):
        if self.creds:
            service = build('drive', 'v3', credentials=self.creds)
            page_token = None
            items = list()
            while True:
                results = service.files()\
                    .list(q=query,
                          pageSize=page_size,
                          pageToken=page_token,
                          fields="nextPageToken, files(id, name)").execute()
                items = items + results.get("files", [])
                page_token = results.get("nextPageToken", None)
                if not page_token:
                    break
            return items

    def download(self, file_id, filename):
        if self.creds:
            service = build('drive', 'v3', credentials=self.creds)
            request = service.files().get_media(fileId=file_id)
            fh = io.FileIO(f"{self.settings['download_directory']}/{filename}",
                           mode="wb")
            downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024)
            pbar = tqdm(total=100)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                pbar.update(int(status.progress() * 100))
            pbar.close()
