WHEN BUILDING A GAME — WHAT MAKES IT FUN AND COMPLETE

A game that just "works" is not enough. It needs to be FUN TO PLAY.
Before writing code, think about what makes the player WANT to keep playing.

CORE GAME DESIGN ELEMENTS:
- CLEAR OBJECTIVE: the player must know what they're trying to do within 5 seconds
- FEEDBACK: every action should have immediate visual/audio feedback
  * Button press → visual change + sound
  * Score increase → animation, not just number change
  * Collision/death → screen shake, flash, particle effect
  * Achievement → celebration animation
- PROGRESSION: the game should get harder or more interesting over time
  * Speed increases, new obstacles, new mechanics unlocked
  * Difficulty curve: easy start → gradual ramp → challenging but fair
- RISK/REWARD: tension between safe play and risky high-reward moves
- JUICE: small details that make actions FEEL satisfying
  * Screen shake on impact
  * Particles on destruction
  * Smooth animations (not instant state changes)
  * Sound effects for every interaction
  * Score popups at point of action

WHAT EVERY GAME NEEDS (don't skip these):
- Start screen with title and "Press to Start" / "Click to Play"
- Score display (visible during gameplay, not hidden)
- High score tracking (at minimum, session best)
- Game over screen with final score and restart option
- Pause functionality (Space or Escape)
- Smooth controls (responsive, not laggy — use requestAnimationFrame)
- Visual polish: consistent art style, even if simple
- Increasing difficulty: the game must get harder over time

COMMON MISTAKES:
- No difficulty curve → boring after 30 seconds
- Instant state changes → no "game feel" (animate everything)
- No feedback on player actions → feels dead
- Controls feel sluggish → frustrating
- No restart button → player has to refresh page
- Game too hard from the start → player quits immediately
- Game too easy forever → no challenge, no reason to play
- No score or goal → no motivation
- Missing edge cases: what happens at screen boundaries, overlapping objects, 
  simultaneous events?

BEFORE WRITING THE PLAN, ASK:
1. What is the core loop? (what does the player do repeatedly?)
2. What makes it HARDER over time?
3. What gives the player FEEDBACK? (visual, audio, score)
4. What makes the player want to try "one more time"?
5. Is there a skill element? (not just luck)
