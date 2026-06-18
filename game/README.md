# Critter Games (AirPoint practice prototype)

A small set of calm, click-to-choose mini-games for young children (around 6) to
practise controlling the cursor with AirPoint. Cross the learning curve by
playing. Single self-contained file: no build, no assets, no dependencies.

## Games (pick from the home page)
- **Find it**: a picture is named; click the matching critter.
- **Match it**: a sample picture is shown; click the one that is the same.
- **Pop it**: a critter bobs around; click it to pop it. Pure aim and click
  practice, no wrong answers (the easiest place to start).
- **Scroll it**: scroll down through a strip of pictures to find the named
  critter, then click it. Practises scrolling.
- **Count it**: count the critters and click the right number. Early numbers.
- **Big or small**: click the big one or the small one. Size words.

## How it plays
- **Click to choose** (not hover), so just moving the cursor never picks
  anything by accident. With AirPoint Kids mode the click is the fist hold.
- **Calm and slow**: big targets, at most 3 choices, generous celebration time,
  a gentle "Try again!" on a wrong pick (no penalty, no timer).
- **Fullscreen**: entering a game requests fullscreen (browsers require a click
  first, so it kicks in on the first tap).
- Stars accumulate (a trophy every 5). The Home button returns to the menu.

## Run it
- **Locally**: open `game/index.html` in any browser.
- **Hosted**: drop the `game/` folder on GitHub Pages, Vercel, or any static
  host and share the URL.

## Cross-platform
Pure web (HTML/CSS/JS), so it behaves the same in Chrome/Edge on **Windows** and
in Safari/Chrome on macOS. Fullscreen uses the standard API with `webkit`/`ms`
fallbacks; input is plain click plus `:hover` with no exotic CSS.

## Tuning
- Difficulty and number of choices: `difficulty()` in the script.
- Picture pool and names: the `POOL` array.
- Celebration pacing: the `setTimeout(..., 1700)` / `1100` delays.

## Pairing with AirPoint
Turn on **Kids mode** in AirPoint (move your hand to move; hold a fist about
1.5s to click; hover a screen edge to scroll), then open this game and pick a
tile. The Scroll it game is a good way to practise the edge hover scroll.
