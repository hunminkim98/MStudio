using System;
using System.ComponentModel;
using System.Threading.Tasks;
using MStudio.Core.Models;

namespace MStudio.Services.Interfaces
{
    public interface ISessionService : INotifyPropertyChanged
    {
        MotionData? CurrentMotion { get; }
        bool IsLoading { get; }
        Task LoadMotionAsync(string filePath);
        void CloseSession();
    }

    public interface ITimelineService : INotifyPropertyChanged
    {
        int CurrentFrame { get; set; }
        int TotalFrames { get; }
        float FrameRate { get; }
        bool IsPlaying { get; set; }
        double PlaybackSpeed { get; set; }
        bool IsLooping { get; set; }
        bool IsReverse { get; set; }
        TimeSpan CurrentTime { get; }
        void Play();
        void Pause();
        void TogglePlay();
        void StepForward();
        void StepBackward();
    }
}
