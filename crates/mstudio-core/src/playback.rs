//! Playback clock. Port of `core/animation_controller.py`, rewritten around
//! plan rule R3: no timers — the UI asks `tick(now)` once per rendered frame
//! and the frame shown is derived from wall-clock time, so playback speed is
//! the data rate regardless of how fast the display refreshes.

use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Playback {
    n_frames: usize,
    fps: f64,
    /// Playback rate multiplier (1.0 = real time).
    pub speed: f64,
    pub looping: bool,
    playing: bool,
    frame: usize,
    /// `(wall time, fractional frame)` at the moment playback (re)started.
    anchor: Option<(Instant, f64)>,
}

impl Default for Playback {
    fn default() -> Self {
        Self { n_frames: 0, fps: 60.0, speed: 1.0, looping: false, playing: false, frame: 0, anchor: None }
    }
}

impl Playback {
    pub fn new() -> Self {
        Self::default()
    }

    /// `AnimationController.set_data_info`.
    pub fn set_data_info(&mut self, n_frames: usize, fps: f64) {
        self.n_frames = n_frames;
        self.fps = if fps > 0.0 { fps } else { 60.0 };
        self.frame = self.frame.min(n_frames.saturating_sub(1));
        self.re_anchor(Instant::now());
    }

    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    pub fn fps(&self) -> f64 {
        self.fps
    }

    pub fn set_fps(&mut self, fps: f64, now: Instant) {
        if fps > 0.0 {
            self.fps = fps;
            self.re_anchor(now);
        }
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }

    pub fn frame(&self) -> usize {
        self.frame
    }

    pub fn play(&mut self, now: Instant) {
        if self.n_frames <= 1 || self.playing {
            return;
        }
        // Playing from the last frame without looping restarts from 0 (Python
        // would stop immediately; restarting is what users expect from ▶).
        if !self.looping && self.frame + 1 >= self.n_frames {
            self.frame = 0;
        }
        self.playing = true;
        self.anchor = Some((now, self.frame as f64));
    }

    pub fn pause(&mut self) {
        self.playing = false;
        self.anchor = None;
    }

    /// Pause and return to the first frame.
    pub fn stop(&mut self) {
        self.pause();
        self.frame = 0;
    }

    pub fn toggle(&mut self, now: Instant) {
        if self.playing {
            self.pause();
        } else {
            self.play(now);
        }
    }

    /// Clamp and jump; keeps playing from the new position if playing.
    pub fn set_frame(&mut self, frame: usize, now: Instant) {
        if self.n_frames == 0 {
            return;
        }
        self.frame = frame.min(self.n_frames - 1);
        self.re_anchor(now);
    }

    pub fn next_frame(&mut self, now: Instant) {
        if self.frame + 1 < self.n_frames {
            self.set_frame(self.frame + 1, now);
        }
    }

    pub fn prev_frame(&mut self, now: Instant) {
        if self.frame > 0 {
            self.set_frame(self.frame - 1, now);
        }
    }

    /// The frame that should be on screen at `now`. Call once per rendered
    /// frame. Stops (or wraps, when looping) at the end.
    pub fn tick(&mut self, now: Instant) -> usize {
        if let Some((t0, f0)) = self.anchor.filter(|_| self.playing) {
            let elapsed = now.saturating_duration_since(t0).as_secs_f64();
            let f = f0 + elapsed * self.fps * self.speed;
            let n = self.n_frames.max(1);
            if self.looping {
                self.frame = (f as usize) % n;
            } else if f >= (n - 1) as f64 {
                self.frame = n - 1;
                self.pause();
            } else {
                self.frame = f as usize;
            }
        }
        self.frame
    }

    pub fn current_time(&self) -> f64 {
        self.frame as f64 / self.fps
    }

    pub fn set_time(&mut self, seconds: f64, now: Instant) {
        let f = (seconds.max(0.0) * self.fps).round() as usize;
        self.set_frame(f, now);
    }

    /// 0.0 … 1.0 along the take.
    pub fn progress(&self) -> f64 {
        if self.n_frames > 1 {
            self.frame as f64 / (self.n_frames - 1) as f64
        } else {
            0.0
        }
    }

    pub fn set_progress(&mut self, progress: f64, now: Instant) {
        let p = progress.clamp(0.0, 1.0);
        let f = if self.n_frames > 1 { (p * (self.n_frames - 1) as f64) as usize } else { 0 };
        self.set_frame(f, now);
    }

    fn re_anchor(&mut self, now: Instant) {
        if self.playing {
            self.anchor = Some((now, self.frame as f64));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn pb(n: usize, fps: f64) -> Playback {
        let mut p = Playback::new();
        p.set_data_info(n, fps);
        p
    }

    #[test]
    fn frame_follows_wall_clock_at_data_rate() {
        let mut p = pb(100, 50.0);
        let t0 = Instant::now();
        p.play(t0);
        assert_eq!(p.tick(t0 + Duration::from_millis(100)), 5);
        assert_eq!(p.tick(t0 + Duration::from_millis(1000)), 50);
        assert!(p.is_playing());
    }

    #[test]
    fn stops_at_last_frame_without_loop_and_wraps_with_loop() {
        let mut p = pb(10, 10.0);
        let t0 = Instant::now();
        p.play(t0);
        assert_eq!(p.tick(t0 + Duration::from_secs(5)), 9);
        assert!(!p.is_playing());

        let mut q = pb(10, 10.0);
        q.looping = true;
        q.play(t0);
        assert_eq!(q.tick(t0 + Duration::from_millis(1250)), 2);
        assert!(q.is_playing());
    }

    #[test]
    fn seeking_while_playing_continues_from_new_frame() {
        let mut p = pb(100, 10.0);
        let t0 = Instant::now();
        p.play(t0);
        p.set_frame(40, t0 + Duration::from_secs(1));
        assert_eq!(p.tick(t0 + Duration::from_millis(1500)), 45);
    }

    #[test]
    fn progress_and_time_round_trip() {
        let mut p = pb(101, 100.0);
        let t0 = Instant::now();
        p.set_progress(0.5, t0);
        assert_eq!(p.frame(), 50);
        assert_eq!(p.current_time(), 0.5);
        p.set_time(0.25, t0);
        assert_eq!(p.frame(), 25);
        p.stop();
        assert_eq!(p.frame(), 0);
        assert!(!p.is_playing());
    }
}
