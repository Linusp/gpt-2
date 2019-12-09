#!/usr/bin/env python3
import os
import re
import json
from datetime import datetime
from collections import defaultdict

import click
import numpy as np
import tensorflow as tf

from src import model, sample, encoder


SENTENCE_PATTERN = re.compile(r'[^.!?]+(?:[.!?]|$)')


def split_sentences(text):
    sents = []
    for sent in SENTENCE_PATTERN.findall(text):
        sent = sent.strip()
        if not sent:
            continue

        if sents and re.match(r'[0-9][\.,]', sents[-1][-2:]) and re.match(r'[0-9]', sent[0]):
            sents[-1] += sent
            continue

        if sents and sents[-1].endswith(' Mr.'):
            sents[-1] = sents[-1] + ' ' + sent
            continue

        sents.append(sent)

    return sents


def postprocess(hint, text):
    text = text.strip()
    if not text:
        return hint
    first_line = text.split('\n')[0]
    sentences = split_sentences(first_line)
    if not sentences:
        return hint

    return hint + ' ' + sentences[0]


def generate(hints, model_name='345M', seed=None,
             nsamples=10, batch_size=1, length=None,
             temperature=1, top_k=0, top_p=1, models_dir='models'):
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))

    batch_size = batch_size or 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    results = defaultdict(set)
    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        for hint in hints:
            print("[%s]begin to generate for: %s" % (datetime.utcnow(), hint))
            context_tokens = enc.encode(hint)
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for out_data in out:
                    text = enc.decode(out_data)
                    text = postprocess(hint, text.strip())
                    results[hint].add(text)

            print("[%s]finished generating for: %s" % (datetime.utcnow(), hint))

    return results


@click.command()
@click.option("-i", "--infile", required=True)
@click.option("-o", "--outfile", required=True)
@click.option("-m", "--model-name", required=True)
@click.option("-d", "--model-dir", required=True)
@click.option("-n", "--nsamples", type=int, default=10)
@click.option("-t", "--temperature", type=float, default=0.7)
def main(infile, outfile, model_name, model_dir, nsamples, temperature):
    hints = [line.strip() for line in open(infile) if line.strip()]
    results = generate(hints, model_name=model_name, models_dir=model_dir,
                       nsamples=nsamples, temperature=temperature)
    with open(outfile, 'w') as fout:
        for hint in hints:
            for new_text in sorted(results[hint]):
                print(f"{hint}\t{new_text}", file=fout)


if __name__ == '__main__':
    main()
