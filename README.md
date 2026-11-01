# PATROL MODE — concept clip

15 s vertical (1080×1920, 30 fps) concept video built on top of the original robot-vacuum footage.
Story: the robot vacuum secretly works as a night home-security guard. **Concept / simulation — not a real product feature.**

- `out/patrol_mode_v2.mp4` — **current**: 16:9, original footage left + iPhone screen recording of the app right
- `out/patrol_mode_v1.mp4` — first version (9:16, HUD over graded footage)
- `render_v2.py` — v2 render; `tools/floor_track.py` + `tools/floor_pose.py` track the floor and solve the camera pose so the standing person stays locked to the floor
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

## v2 layout (1920×1080)
- Left: original footage, no grade. 0–2.2 s = source 0–1.1 s at 0.5x (motion-interpolated): a person (dark legs silhouette) stands still at the doorway next to the robot. Then source 4.0 s onward at normal speed (the camera walks on; the spot is behind it).
- Right: iPhone screen recording of the Patrol app: PATROL MODE, live LiDAR map builds, IR cam detects `PERSON · 94%` (1.2 s), banner `Motion detected · Living room · 02:14` (2.3 s), tap → alert screen with snapshot + live map + robot en route.
- 12.2–15 s: end card `Your vacuum already knows your home.`
```bash
python3 render_v2.py SOURCE.mp4 tools out/patrol_mode_v2.mp4
```
