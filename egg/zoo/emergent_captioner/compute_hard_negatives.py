# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser

import clip
import torch
import tqdm

from dataloaders import (
    CocoWrapper,
    ConceptualCaptionsWrapper,
    FlickrWrapper,
)


def get_opts():
    parser = ArgumentParser()
    parser.add_argument("--output_file_prefix", required=True)
    parser.add_argument(
        "--clip_model",
        choices="ViT-B/16 ViT-B/32".split(),
        default="ViT-B/32",
    )
    parser.add_argument(
        "--dataset",
        choices="coco conceptual flickr".split(),
        default="coco",
    )
    parser.add_argument("--split", choices="train test".split(), default="train")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_negatives", type=int, default=100, help="Max Negatives")

    return parser.parse_args()


def get_dataloader(opts):
    name2wrapper = {
        "conceptual": ConceptualCaptionsWrapper,
        "coco": CocoWrapper,
        "flickr": FlickrWrapper,
    }

    wrapper = name2wrapper[opts.dataset]()

    data_kwargs = dict(
        batch_size=opts.batch_size,
        image_size=opts.image_size,
        num_workers=opts.num_workers,
    )
    loader = wrapper.get_split(split=opts.split, **data_kwargs)
    return loader


def main():
    opts = get_opts()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader = get_dataloader(opts)

    clip_model = clip.load(opts.clip_model)[0].eval().to(device)

    if len(dataloader.dataset) % opts.batch_size != 0:
        print(
            f"Dataset is not divisible by batch size, dropping {len(dataloader.dataset) % opts.batch_size} samples"
        )

    emb_n = (len(dataloader.dataset) // opts.batch_size) * opts.batch_size
    assert emb_n > opts.max_negatives + 1
    emb_s = clip_model.visual.output_dim
    prec_emb = torch.zeros(emb_n, emb_s, dtype=torch.float32, device="cpu")
    prec_nns = torch.zeros(
        emb_n, opts.max_negatives + 1, dtype=torch.int32, device="cpu"
    )

    i = 0
    for batch in tqdm.tqdm(dataloader, desc="Embedding..."):
        images, *_ = batch
        images = images.to(device)

        with torch.no_grad():
            feats = clip_model.encode_image(images)
            feats /= feats.norm(dim=-1, keepdim=True)
            feats = feats.to("cpu")
            prec_emb[i : i + feats.size(0)] = feats
            i += feats.size(0)

    for chunk_start in tqdm.tqdm(
        range(0, emb_n, 1000), desc="Computing nearest neighbours..."
    ):
        chunk = prec_emb[chunk_start : chunk_start + 1000]
        # emb_n x 1000
        cosin = prec_emb @ chunk.t()
        # max_negatives + 1 x 1000
        nns = torch.topk(
            cosin, k=opts.max_negatives + 1, dim=0, largest=True, sorted=True
        ).indices.t()
        # 1000 x max_negatives + 1
        prec_nns[chunk_start : chunk_start + 1000] = nns

    torch.save(prec_emb, opts.output_file_prefix + ".emb.pt")
    torch.save(prec_nns, opts.output_file_prefix + ".nns.pt")


if __name__ == "__main__":
    main()
