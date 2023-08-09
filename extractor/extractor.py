import requests
import json
import time
import os

# DATA_LAKE_PATH = "."
DATA_LAKE_PATH = "/opt/datalake/"
SLEEP_TIME = 30

MASTODON_SERVERS = [
    "https://mastodon.social/api/v1/timelines/public",
    "https://fosstodon.org/api/v1/timelines/public"
]

def fetch_and_save(server, path):
    response = requests.get(server)
    data = response.json()
    filename = f"{path}{int(time.time())}_timeline.json"
    with open(filename, "w") as f:
        json.dump(data, f)

def main():
    os.makedirs(DATA_LAKE_PATH, exist_ok=True)
    while True:
        for server in MASTODON_SERVERS:
            fetch_and_save(server, DATA_LAKE_PATH)
        time.sleep(SLEEP_TIME)

if __name__ == "__main__":
    main()