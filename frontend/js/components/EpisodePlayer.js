/**
 * EpisodePlayer.js — Playback control loop for trajectory frames
 */

export class EpisodePlayer {
  /**
   * @param {object} trajectoryData  Loaded trajectory JSON data {frames, total_steps}
   * @param {function} onFrame        Callback(frameData, currentStep, totalSteps)
   */
  constructor(trajectoryData, onFrame) {
    this.frames = trajectoryData.frames || [];
    this.totalSteps = this.frames.length;
    this.onFrame = onFrame;
    this.currentStep = 0;
    this.isPlaying = false;
    this.speed = 1.0;
    this.timer = null;

    if (this.frames.length > 0) {
      this.onFrame(this.frames[0], 0, this.totalSteps);
    }
  }

  play() {
    if (this.isPlaying) return;
    this.isPlaying = true;
    this._tick();
  }

  pause() {
    this.isPlaying = false;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  toggle() {
    if (this.isPlaying) this.pause();
    else this.play();
  }

  setSpeed(speedMultiplier) {
    this.speed = parseFloat(speedMultiplier);
  }

  seek(stepIndex) {
    this.currentStep = Math.max(0, Math.min(stepIndex, this.totalSteps - 1));
    if (this.frames[this.currentStep]) {
      this.onFrame(this.frames[this.currentStep], this.currentStep, this.totalSteps);
    }
  }

  next() {
    this.seek(this.currentStep + 1);
  }

  prev() {
    this.seek(this.currentStep - 1);
  }

  _tick() {
    if (!this.isPlaying) return;

    if (this.currentStep >= this.totalSteps - 1) {
      this.currentStep = 0; // Loop around
    } else {
      this.currentStep++;
    }

    if (this.frames[this.currentStep]) {
      this.onFrame(this.frames[this.currentStep], this.currentStep, this.totalSteps);
    }

    const interval = Math.max(20, 150 / this.speed);
    this.timer = setTimeout(() => this._tick(), interval);
  }

  destroy() {
    this.pause();
  }
}
