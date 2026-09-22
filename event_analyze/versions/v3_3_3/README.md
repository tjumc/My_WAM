# V3.3.3 — Object-Centric Episodes + Transition Completion

V3.3.3 keeps the V3.3.2 proposal/Qwen observations fixed by default and changes
only temporal reasoning, so improvements can be causally attributed.

## Main changes

1. **Object-centric manipulation episodes**
   - placement skills are formed from an object's state trajectory before skill-list deduplication;
   - short target contacts followed by a quick re-grasp remain inside the same episode;
   - unrelated rack/basket candidates no longer prevent two parts of the same object manipulation from being grouped.

2. **Dense motion with weak endpoint completion**
   - `moving_out` supports a later `out` endpoint;
   - `moving_in` supports a later `in` endpoint;
   - this is a weak transition prior, not a direct absolute-state observation, so generic visual evidence can override it.

3. **Initial lifecycle from early direct observations**
   - lifecycle initialization no longer reads the first Viterbi segment;
   - only the first few generic Qwen observations are used;
   - early directional observations such as `opening` vote for their physical source state (`closed`).

4. **Physical accessibility prerequisite**
   - dish-rack and cutlery-basket interactions require the dishwasher door lifecycle to be open;
   - door temporal spans are not shifted later to a contact anchor.

## Controlled run

```bash
bash versions/v3_3_3/run.sh \
  /path/to/episode.hdf5 \
  output/episode_analysis \
  "put the dish into the dishwasher"
```

If V3.3.2 results exist, proposal and Qwen observations are reused automatically.
Use `V333_REPROPOSE=1`, `V333_REOBSERVE=1`, or `V333_REDENSE=1` only when an
intentional fresh perception run is desired.
