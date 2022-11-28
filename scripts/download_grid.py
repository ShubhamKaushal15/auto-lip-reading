from bs4 import BeautifulSoup
import requests, zipfile, io, os
from os.path import expanduser

home = expanduser("~")

def download_and_extract(download_url, # change path to save
                         path_to_save = os.path.join(home, "grid_videos")):

    if not os.path.exists(path_to_save):
        print(f"path did not exist. Creating {path_to_save}...")
        os.makedirs(path_to_save)

    r = requests.get(download_url)
    if not r.ok:
        print(f"Status not OK for {download_url}")
        return
        
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path_to_save)

def main():

    base_URL = "https://spandh.dcs.shef.ac.uk//gridcorpus/"
    download_suffix = "#downloads"
    page = requests.get(base_URL + download_suffix)
    soup = BeautifulSoup(page.content, "html.parser")

    all_data_table = soup.find_all("table")[2]
    table_rows = all_data_table.find_all("tr")

    video_download_suffixes = []
    column_index = 2
    for i in range(len(table_rows)):
        if i == 0 or i == 21:
            continue
        video_download_suffixes.append(table_rows[i].find_all("a")[column_index].get("href"))

    for suffix in video_download_suffixes:
        print(suffix)
        download_and_extract(base_URL + suffix)
        print("OK")

if __name__ == '__main__':
    main()