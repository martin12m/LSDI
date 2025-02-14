from kaggle.api.kaggle_api_extended import KaggleApi
import os
import time

def kaggleDatasets(max_datasets=2000, min_size=20e3, max_size=2e6, batch_size=50, sleep_time=10):
    api = KaggleApi()
    api.authenticate()

    download_folder = "kaggle_files"
    os.makedirs(download_folder, exist_ok=True)

    downloaded_datasets = 0
    page = 1
    downloaded_files = []

    while downloaded_datasets < max_datasets:
        datasets = api.dataset_list(sort_by="hottest", page=page)

        if not datasets:
            print("No more datasets available")
            break

        for dataset in datasets:
            if downloaded_datasets >= max_datasets:
                print("Reached the maximum dataset limit")
                break

            files = api.dataset_list_files(dataset.ref).files
            if not files:
                continue

            for file in files:
                if not file.name.endswith(".csv"):
                    continue

                file_size = file.totalBytes
                if not (min_size <= file_size <= max_size):
                    continue

                file_path = os.path.join(download_folder, file.name)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                try:
                    api.dataset_download_file(
                        dataset=dataset.ref,
                        file_name=file.name,
                        path=os.path.dirname(file_path),
                        force=False,
                        quiet=False
                    )
                    downloaded_files.append(file_path)
                    downloaded_datasets += 1

                    if downloaded_datasets >= max_datasets:
                        print("Reached the maximum dataset limit")
                        break
                except Exception as e:
                    print(f"Failed to download {file.name}: {e}")

            if downloaded_datasets % batch_size == 0:
                #print(f"Sleeping for {sleep_time} seconds to avoid hitting API rate limit...")
                time.sleep(sleep_time)

        page += 1

kaggleDatasets(max_datasets=15000, max_size=2e6, min_size=20e3)
