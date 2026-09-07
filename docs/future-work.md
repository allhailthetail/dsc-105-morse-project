# Future Work (not built yet)

These are deliberately **not implemented** in this phase — captured here as
design notes so the decisions aren't lost, per the user's own framing of them
as "in the future" work, distinct from the real-time console decoder
(`scripts/run_realtime.py`), which *is* built.

All three below should consume the **same underlying "decoded message" event
source** — whatever `scripts/run_realtime.py` already produces each time it
prints a finalized message — rather than each reimplementing capture/decode
independently. Concretely, that likely means: `run_realtime.py` grows a
pluggable "sink" (currently just `print()`) that can also write to a log/DB,
and the web UI and display both read from that log/DB rather than running
their own copy of the segmenter+model.

## Lightweight web UI

Serve the decoded-message stream through a small Flask/FastAPI page: a
live-updating list of received messages (poll or a simple SSE/WebSocket
push), likely reading from the persistent log described below rather than
holding messages only in the decoder process's memory.

## Persistent logging

Append each finalized message with a timestamp to a local file or SQLite DB
as it's decoded. This is the natural shared source for both the web UI and
the display — build this first, once it's time to build any of the three.

## 3.5" 320x480 TFT touch display

Show received messages, updating live as new ones arrive.

Notes for whoever picks this up: Pi-attached SPI TFTs like this commonly need
a kernel framebuffer driver installed (e.g. a "goodtft/LCD-show"-style
overlay) before any GUI toolkit (pygame/Kivy/Qt) can draw to them at all —
the exact driver/setup depends on the display's actual controller chip (often
ILI9486 or similar for this size/resolution, but **confirm against the real
hardware** rather than assuming). Once the framebuffer driver is working, a
simple scrolling-text pygame or Kivy app reading from the persistent log
above is the most direct implementation — touch input isn't needed for a
read-only "show incoming messages" display, so it can be ignored unless a
future feature (e.g. adjusting silence threshold from the device itself)
calls for it.
