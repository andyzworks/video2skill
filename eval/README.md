# Video2Skill evaluation

Scoring code for the Video2Skill benchmark ([paper](https://arxiv.org/abs/2609.36691),
[data](https://huggingface.co/datasets/Sterzhang/video2skill-bench)). Python 3.8+, standard library only.

## Reproduce Table 1

`predictions/` holds the outputs of the 19 models in Table 1 (unified and factorized, RoboInter and HD-EPIC).
Rescoring them prints the table exactly as in the paper:

```bash
pip install huggingface_hub   # only to fetch the test files; or pass --data <local copy>
python reproduce_main_table.py
```

## Score your own model

```bash
python score.py --test <data>/test/hdepic.jsonl --pred my_model_hdepic.jsonl
# Cov. 44.1  Pair P 13.1  Pair R 48.9  ARI 11.5  Create R 12.2  Reuse R 97.2
```

**Prediction file.** One line per test clip:

```json
{"clip_id": "he__P04-20240414-162750__c1", "events": [{"start": 16.0, "end": 21.0, "skill": "place_object(object, destination)"}]}
```

- `start` and `end` are seconds from the start of the clip.
- `skill` is the model's own skill name. Only the part before `(` is used, lower-cased. Names are free-form: the
  metrics only look at which events share a skill, not at the names themselves.
- A clip with no line is scored as having no predictions.

**Protocol.** The model must process the clips in the order of the test file, carrying its skill library from
one clip to the next. The library starts empty. The model never sees the reference class list.

**Metrics** (Section 2.3 of the paper). Each reference event is matched to the prediction that overlaps it most in
time, if their tIoU is at least 0.3. Unmatched predictions are not penalized.

| Metric | Meaning |
|---|---|
| Cov. | Share of reference events that are matched |
| Pair P | Of matched event pairs the model puts in one skill, the share that are the same reference class. Low = different actions mixed together |
| Pair R | Of matched event pairs in the same reference class, the share the model puts in one skill. Low = one action split into many skills |
| ARI | Adjusted Rand index between the model's skills and the reference classes. 0 = chance |
| Create R | First time a reference class appears: share that get a skill name not used before |
| Reuse R | Later appearances of a reference class: share that get a skill name used before |

Pair P, Pair R, ARI, Create R and Reuse R are computed over matched events only.

## Inference settings used in the paper

- **Video input.** Each clip is split into 16-second chunks sampled at 1 frame per second (16 frames); a short
  remainder is merged into the last chunk. Frames are resized to 448 pixels wide and given to the model one image
  per frame, each preceded by `Frame k at t s:`.
- **Unified.** At every chunk the model sees the frames, the current skill library (up to 80 skills, each with its
  schema, definition, usage count and three recent examples) and any unfinished event from the previous chunk. It
  returns new skills, completed events and an unfinished event. The library persists across clips.
- **Factorized.** The model first proposes events from the frames without seeing the library. The same model then
  updates the library from the text of those events, chunk by chunk, without seeing the video.
- **Decoding.** Greedy, with Hugging Face `transformers`. Up to 512 new tokens per video chunk (2,048 for
  GLM-4.1V-9B) and 384 per text-only library update.
