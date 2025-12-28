using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Timers;
using MStudio.Services.Interfaces;

namespace MStudio.Services.Implementations
{
    public class TimelineService : ITimelineService, IDisposable
    {
        private readonly ISessionService _sessionService;
        private readonly ITrialService? _trialService;
        private readonly System.Timers.Timer _timer;
        private readonly SynchronizationContext? _syncContext;
        private int _currentFrame;
        private bool _isPlaying;
        private double _playbackSpeed = 1.0;
        private bool _isLooping;
        private bool _isReverse;
        private double _accumulatedFrame;

        // Primary constructor with TrialService support
        public TimelineService(ISessionService sessionService, ITrialService trialService)
            : this(sessionService)
        {
            _trialService = trialService;
            _trialService.SelectionChanged += OnTrialSelectionChanged;
            _trialService.PropertyChanged += OnTrialServicePropertyChanged;
        }

        // Legacy constructor for backward compatibility
        public TimelineService(ISessionService sessionService)
        {
            _sessionService = sessionService;
            _sessionService.PropertyChanged += OnSessionChanged;

            _syncContext = SynchronizationContext.Current;

            _timer = new System.Timers.Timer();
            _timer.Elapsed += OnTimerElapsed;
            _timer.AutoReset = true;
        }

        public int CurrentFrame
        {
            get => _currentFrame;
            set
            {
                int max = TotalFrames > 0 ? TotalFrames - 1 : 0;
                int clamped = Math.Clamp(value, 0, max);
                if (SetProperty(ref _currentFrame, clamped))
                {
                    _accumulatedFrame = clamped;
                    OnPropertyChanged(nameof(CurrentTime));
                }
            }
        }

        /// <summary>
        /// Total frames based on selected trials (if available) or current session motion.
        /// When multiple trials are selected, uses the maximum frame count.
        /// </summary>
        public int TotalFrames => 
            _trialService?.HasSelectedTrials == true 
                ? _trialService.MaxSelectedFrameCount 
                : _sessionService.CurrentMotion?.Metadata.TotalFrames ?? 0;

        /// <summary>
        /// Frame rate based on selected trials (if available) or current session motion.
        /// When multiple trials are selected, uses the maximum frame rate.
        /// </summary>
        public float FrameRate => 
            _trialService?.HasSelectedTrials == true 
                ? _trialService.MaxSelectedFrameRate 
                : _sessionService.CurrentMotion?.Metadata.FrameRate ?? 30.0f;
        
        public TimeSpan CurrentTime
        {
            get
            {
                if (FrameRate <= 0) return TimeSpan.Zero;
                return TimeSpan.FromSeconds(CurrentFrame / FrameRate);
            }
        }

        public bool IsPlaying
        {
            get => _isPlaying;
            set
            {
                if (SetProperty(ref _isPlaying, value))
                {
                    if (_isPlaying) _timer.Start();
                    else _timer.Stop();
                }
            }
        }

        public double PlaybackSpeed
        {
            get => _playbackSpeed;
            set
            {
                if (SetProperty(ref _playbackSpeed, Math.Max(0, value)))
                {
                    UpdateTimerInterval();
                }
            }
        }

        public bool IsLooping
        {
            get => _isLooping;
            set => SetProperty(ref _isLooping, value);
        }

        public bool IsReverse
        {
            get => _isReverse;
            set => SetProperty(ref _isReverse, value);
        }

        private void OnSessionChanged(object? sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(ISessionService.CurrentMotion))
            {
                CurrentFrame = 0;
                OnPropertyChanged(nameof(TotalFrames));
                OnPropertyChanged(nameof(FrameRate));
                OnPropertyChanged(nameof(CurrentTime));
                UpdateTimerInterval();
            }
        }

        private void OnTrialSelectionChanged(object? sender, EventArgs e)
        {
            // When trial selection changes, update timeline properties
            CurrentFrame = Math.Min(CurrentFrame, TotalFrames > 0 ? TotalFrames - 1 : 0);
            OnPropertyChanged(nameof(TotalFrames));
            OnPropertyChanged(nameof(FrameRate));
            OnPropertyChanged(nameof(CurrentTime));
            UpdateTimerInterval();
        }

        private void OnTrialServicePropertyChanged(object? sender, PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(ITrialService.MaxSelectedFrameCount) ||
                e.PropertyName == nameof(ITrialService.MaxSelectedFrameRate))
            {
                OnPropertyChanged(nameof(TotalFrames));
                OnPropertyChanged(nameof(FrameRate));
                UpdateTimerInterval();
            }
        }

        private void UpdateTimerInterval()
        {
            if (FrameRate > 0)
            {
                _timer.Interval = 1000.0 / FrameRate;
            }
        }

        private void OnTimerElapsed(object? sender, ElapsedEventArgs e)
        {
            if (TotalFrames <= 1) return;

            // Calculate direction and speed
            double speed = _playbackSpeed * (_isReverse ? -1 : 1);
            
            _accumulatedFrame += speed;

            int nextFrame = (int)Math.Floor(_accumulatedFrame);
            int maxFrame = TotalFrames - 1;

            if (speed > 0)
            {
                if (nextFrame > maxFrame)
                {
                    if (_isLooping)
                    {
                        nextFrame = 0;
                        _accumulatedFrame = 0;
                    }
                    else
                    {
                        nextFrame = maxFrame;
                        _accumulatedFrame = maxFrame;
                        // Optional: Pause at end if not looping
                         // IsPlaying = false; // Note: Setting this from wrong thread requires marshalling if we want UI update, but let's keep it simple
                    }
                }
            }
            else if (speed < 0)
            {
                if (nextFrame < 0)
                {
                    if (_isLooping)
                    {
                        nextFrame = maxFrame;
                        _accumulatedFrame = maxFrame;
                    }
                    else
                    {
                        nextFrame = 0;
                        _accumulatedFrame = 0;
                    }
                }
            }

            if (nextFrame != _currentFrame)
            {
                _currentFrame = nextFrame;
                OnPropertyChangedThreadSafe(nameof(CurrentFrame));
                OnPropertyChangedThreadSafe(nameof(CurrentTime));
            }
        }

        public void Play() => IsPlaying = true;
        public void Pause() => IsPlaying = false;
        public void TogglePlay() => IsPlaying = !IsPlaying;
        public void StepForward() { CurrentFrame++; Pause(); }
        public void StepBackward() { CurrentFrame--; Pause(); }

        public event PropertyChangedEventHandler? PropertyChanged;
        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        private void OnPropertyChangedThreadSafe(string propertyName)
        {
            if (_syncContext != null)
            {
                _syncContext.Post(_ => OnPropertyChanged(propertyName), null);
            }
            else
            {
                OnPropertyChanged(propertyName);
            }
        }

        protected bool SetProperty<T>(ref T storage, T value, [CallerMemberName] string? propertyName = null)
        {
            if (EqualityComparer<T>.Default.Equals(storage, value)) return false;
            storage = value;
            OnPropertyChanged(propertyName);
            return true;
        }

        public void Dispose()
        {
            _timer.Stop();
            _timer.Dispose();
            _sessionService.PropertyChanged -= OnSessionChanged;
            if (_trialService != null)
            {
                _trialService.SelectionChanged -= OnTrialSelectionChanged;
                _trialService.PropertyChanged -= OnTrialServicePropertyChanged;
            }
        }
    }
}
