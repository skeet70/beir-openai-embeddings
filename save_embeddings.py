import requests
from tqdm.autonotebook import tqdm
import os
import json
import pathlib
import zipfile
import tiktoken


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


def process_chunks(chunks, embeddings_file):
    response = requests.request(
        method="post",
        url="https://api.openai.com/v1/embeddings",
        json={"model": embedding_model, "input": chunks},
        headers={
            "Authorization": "Bearer {}".format(os.environ["OPENAI_API_KEY"]),
            "Content-Type": "application/json",
        },
    )
    json = response.json()
    if "data" in json:
        embeddings = response.json()["data"]
        chunk_embeddings = np.array(
            [embedding_object["embedding"] for embedding_object in embeddings]
        )
        full_doc_embedding = np.mean(chunk_embeddings, axis=0)

        embeddings_file.writeline(
            [
                "{}\n".format(embedding_object["embedding"])
                for embedding_object in embeddings
            ]
        )
    else:
        print(json)
        print(chunks)
        print("Failed a batch, trying again. This is an infinite loop.")
        process_chunks(chunks, embeddings_file)


def count_existing_lines(file_path):
    """Count the number of lines in an existing embeddings file."""
    try:
        with open(file_path, "r") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0  # If file doesn't exist, start from the beginning


# ada-002 has a limit of 8192 tokens, so if we're coming close to that, break the text up into multiple chunks
# that can be averages (naively)
def chunk_text(text, max_tokens=7500):
    encoding = toktoken.get_encoding(
        "cl100k_base"
    )  # the encoder used for ada-002 according to their docs
    tokens = encoding.encode(text)
    chunks = [tokens[x : x + max_tokens] for x in range(0, len(tokens), max_tokens)]
    return [encoding.decode(x) for x in chunks]


def get_embeddings(source_path, output_path):
    existing_line_count = count_existing_lines(output_path)
    source_line_count = sum(1 for _ in open(source_path, "r"))
    with open(source_path) as source_file, open(output_path, "a+") as embeddings_file:
        # skip already processed lines
        for _ in range(existing_line_count):
            next(source_file, None)

        # process the remaining lines
        for source_line in tqdm(
            source_file, total=source_line_count - existing_line_count
        ):
            source_json = json.loads(source_line)
            chunks = chunk_text(source_json["title"] + " " + source_json["text"])
            process_chunks(chunks, embeddings_file)


# change to get embeddings for a different dataset.
dataset = "dbpedia-entity"
# change to use a different openai model
embedding_model = "text-embedding-ada-002"

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
get_embeddings(queries_path, queries_embedding_path)
print("Generating corpus embeddings for {} using {}.".format(dataset, embedding_model))
get_embeddings(corpus_path, corpus_embedding_path)
