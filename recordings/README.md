# Real recordings (`real` test set)

This folder holds short real-world clips recorded by the author. They form the `real` evaluation set (PLAN.md §4.10.3). There is no clean reference for them, so they are scored with DNSMOS P.835 only.

## What to record

- **How many:** 20–30 clips.
- **Length:** 8–15 s each. Start talking within the first second and keep talking for most of the clip.
- **Devices:** use both of the following, and say which one in the manifest:
  - an **earbud / headset mic** (AirPods or similar, connected to a phone or laptop);
  - the **phone's built-in mic** (held normally, or on speaker in your hand).
  - A laptop mic is a useful extra if you have spare time.
- **Places:** spread the clips over these, with at least 3 per place:
  - café or restaurant;
  - street or traffic;
  - office or open-plan room (keyboards, colleagues, air conditioning);
  - outdoors (park, wind, birds; avoid strong direct wind on the mic);
  - car (engine running, driving if a passenger records, or parked with the engine on).
- **Speech:** speak naturally at normal call volume, as if you were on a call. Read a short paragraph or just talk; vary it between clips. Don't shout over the noise unless that's what you would really do.
- **Mix of conditions:** include a few clips with a quiet moment (2–3 s of pause), one or two with another person talking nearby, and one that's quite loud.
- **Format:** `.wav` or `.m4a` (the iPhone Voice Memos default) are both fine, at any sample rate, mono or stereo. The build script converts everything to 16 kHz mono. Don't apply any noise reduction or "voice isolation" while recording. On iPhone, turn off Voice Memos → *Enhance Recording*, and set Control Centre → *Mic Mode* to *Standard*.

## Consent and privacy

- Only record **your own voice**. If anyone else is clearly audible (for example a friend in the car), ask them first and note `consent: yes` in `notes`. Otherwise re-record.
- Avoid saying names, addresses, phone numbers or anything private. These clips are committed to the repo and may be played on the website.
- Background chatter from strangers in public places should be unintelligible. If someone nearby is clearly intelligible, re-record somewhere else.

## Adding clips

1. Put the audio files in this folder, e.g. `recordings/cafe_earbud_01.m4a`.
2. Add one row per clip to `recordings/manifest.csv`:

   ```csv
   id,file,device,place,notes
   cafe_earbud_01,cafe_earbud_01.m4a,earbud,cafe,"busy lunchtime, espresso machine"
   street_phone_01,street_phone_01.wav,phone,street,"bus passing at ~6 s"
   ```

   | column  | meaning |
   |---------|---------|
   | `id`     | unique, lowercase, no spaces; becomes `data/real/<id>.wav` |
   | `file`   | path relative to `recordings/` |
   | `device` | `earbud`, `phone`, `laptop`, … |
   | `place`  | `cafe`, `street`, `office`, `outdoors`, `car`, … (used as the noise-type label) |
   | `notes`  | anything useful: noise events, distance to mic, `consent: yes`, … |

3. Rebuild the set:

   ```bash
   uv run scripts/build_ood_sets.py --out data --seed 0 --only real
   ```

   This writes `data/real/<id>.wav` (16 kHz mono PCM16) and `data/manifests/real.csv`. It warns about clips outside 8–15 s. If the manifest has no rows, it prints a note and skips.
