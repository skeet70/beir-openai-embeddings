# beir-openai-embeddings

## pregenerated

Previously generated datasets are available in `/datasets`. [Git LFS](https://git-lfs.com/) is required to access them.

### usage

Extract the `tar.gz` file of the dataset you want to use. The resulting `*-embeddings.jsonl` files are the pregenerated embeddings, one embedding per line. You'll need to load those into BeIR yourself.

## generation

If you'd like to use embeddings for a dataset that doesn't exist in the repository yet, you can generate them with the script `save_embeddings.py`.

It'd be appreciated if you `.tar.gz` the resulting `*-embeddings.jsonl` files and make a PR adding them to the repository using [Git LFS](https://git-lfs.com/) so everyone can benefit. The file naming format is `{dataset}-{embedding-model}.tar.gz`.

### requirements

* requests
* tqdm

If you have nix flakes/direnv enabled, there's a `flake.nix` that'll provide the requirements for you.

### usage

Set `OPENAI_API_KEY` in your environment, or put it in a file for later expansion with `env`.

Set the configurable variables in the script to your desired values:

* dataset - the name of the [BeIR datasets](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/) you'd like to create embeddings for.
* embedding_model - the [OpenAI embedding model](https://platform.openai.com/docs/guides/embeddings/embedding-models) you want to use to generate the embeddings
* BATCH_SIZE - the size of batched text calls to OpenAI's embedding endpoint. Larger sizes will mean less time spent making network calls and help prevent getting rate limited by OpenAI. They'll also mean larger response bodies though, adjust to your desired balance.

Run the python script:

```console
env $(cat .openai) python save_embeddings.py
```

## development

Use the provided `flake.nix` to get a working development environment. Before comitting, make sure to run `black ./` and `ruff --fix ./` to fix style and linting issues.

You'll need to run `git lfs install --local` to enable `git lfs` if comitting pregenerated datasets.
