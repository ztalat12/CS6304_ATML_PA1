"""Download PACS (same source as DomainBed: facebookresearch/DomainBed/domainbed/scripts/download.py).
Result: <root>/PACS/{art_painting,cartoon,photo,sketch}/<class>/<images>
If the Google-Drive link is rate-limited, download PACS manually (the 'kfold' folder)
and rename it to <root>/PACS.
"""
import argparse
import os
import zipfile

URL = "https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd"   # DomainBed's PACS link

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    root = ap.parse_args().root
    os.makedirs(root, exist_ok=True)
    import gdown
    zpath = os.path.join(root, "PACS.zip")
    gdown.download(URL, zpath, quiet=False)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    os.rename(os.path.join(root, "kfold"), os.path.join(root, "PACS"))
    os.remove(zpath)
    print("PACS ready at", os.path.join(root, "PACS"))
