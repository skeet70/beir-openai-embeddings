import requests
from tqdm.autonotebook import tqdm
import os
import json
import pathlib
import zipfile


def download_url(url: str, save_path: str, chunk_size: int = 1024):
    """Download url with progress bar using tqdm
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

    Args:
        url (str): downloadable url
        save_path (str): local path to save the downloaded file
        chunk_size (int, optional): chunking of files. Defaults to 1024.
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get("Content-Length", 0))
    with open(save_path, "wb") as fd, tqdm(
        desc=save_path,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=chunk_size,
    ) as bar:
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)


def download_and_unzip(url: str, out_dir: str, chunk_size: int = 1024) -> str:
    os.makedirs(out_dir, exist_ok=True)
    dataset = url.split("/")[-1]
    zip_file = os.path.join(out_dir, dataset)

    if not os.path.isfile(zip_file):
        download_url(url, zip_file, chunk_size)

    if not os.path.isdir(zip_file.replace(".zip", "")):
        zip_ = zipfile.ZipFile(zip_file, "r")
        zip_.extractall(path=out_dir)
        zip_.close()

    return os.path.join(out_dir, dataset.replace(".zip", ""))


def process_batch(batch, embeddings_file):
    response = requests.request(
        method="post",
        url="https://api.openai.com/v1/embeddings",
        json={"model": embedding_model, "input": batch},
        headers={
            "Authorization": "Bearer {}".format(os.environ["OPENAI_API_KEY"]),
            "Content-Type": "application/json",
        },
    )
    embeddings = response.json()["data"]
    embeddings_file.writelines(
        [
            "{}\n".format(embedding_object["embedding"])
            for embedding_object in embeddings
        ]
    )


def get_embeddings(source_path, output_path, batch_size):
    source_line_count = sum(1 for line in open(source_path, "r"))
    with open(source_path) as source_file, open(output_path, "a+") as embeddings_file:
        batch = []
        # clear the output file
        embeddings_file.seek(0)
        embeddings_file.truncate()
        embeddings_file.flush()
        for source_line in tqdm(source_file, total=source_line_count):
            source_json = json.loads(source_line)
            batch.append(source_json["text"])
            if len(batch) == batch_size:
                process_batch(batch, embeddings_file)
                batch = []
        process_batch(batch, embeddings_file)


# change to get embeddings for a different dataset.
dataset = "nfcorpus"
# change to use a different openai model
embedding_model = "text-embedding-ada-002"
# change to make larger or smaller calls to openAI. Balance speed and response size.
BATCH_SIZE = 100

# download the BeIR dataset
url = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset
    )
)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = download_and_unzip(url, out_dir)
queries_path = data_path + "/queries.jsonl"
queries_embedding_path = data_path + "/queries-{}-embeddings.jsonl".format(
    embedding_model
)
corpus_path = data_path + "/corpus.jsonl"
corpus_embedding_path = data_path + "/corpus-{}-embeddings.jsonl".format(
    embedding_model
)

print("Generating query embeddings for {} using {}.".format(dataset, embedding_model))
get_embeddings(queries_path, queries_embedding_path, BATCH_SIZE)
print("Generating corpus embeddings for {} using {}.".format(dataset, embedding_model))
get_embeddings(corpus_path, corpus_embedding_path, BATCH_SIZE)
