# PATROL MODE — concept clip

15 s vertical (1080×1920, 30 fps) concept video built on top of the original robot-vacuum footage.
Story: the robot vacuum secretly works as a night home-security guard. **Concept / simulation — not a real product feature.**

- `out/patrol_mode_v1.mp4` — rendered clip
- `render.py` — the whole render (Python + numpy + Pillow, piped into ffmpeg)
- `tools/track.py` — finds the robot in the source video (dark-blob tracker) → `tools/track.json`

## Timeline
| time | shot |
|---|---|
| 0–3 s | source 9–12 s, cold-blue night grade, HUD `PATROL MODE · 02:14`, reticle on the robot |
| 3–6 s | source 14–17 s, apartment point cloud + wireframe grows from the robot, rotating LiDAR beam, rooms fill in |
| 6–9 s | synthetic floor-level robot POV (IR look), silhouette walks in, `PERSON · 94%` box + thermal edge |
| 9–12 s | glitch → lock screen, `Motion detected · Living room · 02:14`, tap, mini map with robot + red alert dot |
| 12–15 s | source 26.4–29.4 s, robot drives toward camera, end card + `Not a real product feature` |

`CONCEPT · SIMULATION` is burned into the top-right corner of every frame.

## Re-render
```bash
python3 tools/track.py SOURCE.mp4            # writes /tmp/work/track.json
python3 render.py SOURCE.mp4 tools/track.json out/patrol_mode_v1.mp4
python3 render.py SOURCE.mp4 tools/track.json /tmp/dbg --frames 30,150,300   # preview PNGs
```
